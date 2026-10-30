import { describe, test, expect } from "bun:test";
import { eq, and } from "drizzle-orm";
import { write, writeRaw } from "../write.js";
import { rm } from "../rm.js";
import { cp } from "../cp.js";
import { mv } from "../mv.js";
import { edit } from "../edit.js";
import { append } from "../append.js";
import { revert } from "../revert.js";
import { cat } from "../cat.js";
import { createVersion } from "../versioning.js";
import type { OpContext } from "../types.js";
import { schema, type DB } from "../../db/index.js";
import { EditConflictError } from "../../errors.js";
import { createTestContext, type MockS3Client } from "../../test-utils.js";

function versionNumbers(db: DB, driveId: string, path: string): number[] {
  return db
    .select({ version: schema.fileVersions.version })
    .from(schema.fileVersions)
    .where(and(eq(schema.fileVersions.path, path), eq(schema.fileVersions.driveId, driveId)))
    .all()
    .map((v) => v.version)
    .sort();
}

/** Hold the first call to `s3[method]` until `resume()`; count every call. */
function pauseFirstCall(
  s3: MockS3Client,
  method: "putObject" | "copyObject" | "deleteObject"
) {
  const original = (s3[method] as (...args: unknown[]) => Promise<unknown>).bind(s3);
  let resume!: () => void;
  const gate = new Promise<void>((resolve) => {
    resume = resolve;
  });
  let paused!: () => void;
  const reached = new Promise<void>((resolve) => {
    paused = resolve;
  });
  const state = { calls: 0, reached, resume };
  (s3 as any)[method] = async (...args: unknown[]) => {
    if (state.calls++ === 0) {
      paused();
      await gate;
    }
    return original(...args);
  };
  return state;
}

describe("expectedVersion check, storage write and commit are atomic per path", () => {
  test("a create-only write paused in putObject is not overtaken by a second one", async () => {
    const { ctx, db, s3 } = createTestContext();
    await write(ctx, { path: "/gone.txt", content: "v1" });
    await rm(ctx, { path: "/gone.txt" }); // head: v2 delete tombstone

    const put = pauseFirstCall(s3, "putObject");
    const first = write(ctx, { path: "/gone.txt", content: "first", expectedVersion: 0 });
    await put.reached;
    const second = write(ctx, { path: "/gone.txt", content: "second", expectedVersion: 0 });
    // Give the second writer every chance to finish while the first is paused.
    await Bun.sleep(0);
    put.resume();
    const [a, b] = await Promise.allSettled([first, second]);

    expect(a.status).toBe("fulfilled");
    expect((a as PromiseFulfilledResult<any>).value.version).toBe(3);
    expect(b.status).toBe("rejected");
    expect((b as PromiseRejectedResult).reason).toBeInstanceOf(EditConflictError);
    // The loser never reached storage, so it cannot clobber the winner's bytes.
    expect(put.calls).toBe(1);
    expect((await cat(ctx, { path: "/gone.txt" })).content).toBe("first");
    expect(versionNumbers(db, ctx.driveId, "/gone.txt")).toEqual([1, 2, 3]);
  });

  test("a create-only cp paused in copyObject holds off a write to the same path", async () => {
    const { ctx, db, s3 } = createTestContext();
    await write(ctx, { path: "/src.txt", content: "copied" });
    await write(ctx, { path: "/dst.txt", content: "v1" });
    await rm(ctx, { path: "/dst.txt" });

    const copy = pauseFirstCall(s3, "copyObject");
    const copied = cp(ctx, { from: "/src.txt", to: "/dst.txt", expectedVersion: 0 });
    await copy.reached;
    const written = write(ctx, { path: "/dst.txt", content: "written", expectedVersion: 0 });
    await Bun.sleep(0);
    copy.resume();
    const [a, b] = await Promise.allSettled([copied, written]);

    expect(a.status).toBe("fulfilled");
    expect(b.status).toBe("rejected");
    expect((b as PromiseRejectedResult).reason).toBeInstanceOf(EditConflictError);
    expect((await cat(ctx, { path: "/dst.txt" })).content).toBe("copied");
    expect(versionNumbers(db, ctx.driveId, "/dst.txt")).toEqual([1, 2, 3]);
  });

  // Every op that checks expectedVersion, then writes storage, then commits:
  // pause its storage call and let a write that expects the same head (v2) race it.
  const racingOps: Array<{
    name: string;
    method: "putObject" | "copyObject" | "deleteObject";
    run: (ctx: OpContext) => Promise<unknown>;
    content: string | null;
  }> = [
    {
      name: "writeRaw",
      method: "putObject",
      run: (ctx) =>
        writeRaw(ctx, { path: "/race.txt", bytes: new TextEncoder().encode("raw"), expectedVersion: 2 }),
      content: "raw",
    },
    {
      name: "edit",
      method: "putObject",
      run: (ctx) =>
        edit(ctx, { path: "/race.txt", old_string: "two", new_string: "edited", expectedVersion: 2 }),
      content: "edited",
    },
    {
      name: "append",
      method: "putObject",
      run: (ctx) => append(ctx, { path: "/race.txt", content: "+", expectedVersion: 2 }),
      content: "two+",
    },
    {
      name: "revert",
      method: "putObject",
      run: (ctx) => revert(ctx, { path: "/race.txt", version: 1, expectedVersion: 2 }),
      content: "one",
    },
    {
      name: "rm",
      method: "deleteObject",
      run: (ctx) => rm(ctx, { path: "/race.txt", expectedVersion: 2 }),
      content: null,
    },
    {
      name: "mv",
      method: "copyObject",
      run: (ctx) => mv(ctx, { from: "/race.txt", to: "/moved.txt", expectedVersion: 2 }),
      content: null,
    },
  ];

  for (const op of racingOps) {
    test(`${op.name} paused in ${op.method} holds off a write that expects the same head`, async () => {
      const { ctx, db, s3 } = createTestContext({ versioningEnabled: true });
      await write(ctx, { path: "/race.txt", content: "one" });
      await write(ctx, { path: "/race.txt", content: "two" });

      const paused = pauseFirstCall(s3, op.method);
      const winner = op.run(ctx);
      await paused.reached;
      const rival = write(ctx, { path: "/race.txt", content: "rival", expectedVersion: 2 });
      await Bun.sleep(0);
      paused.resume();
      const [a, b] = await Promise.allSettled([winner, rival]);

      expect(a.status).toBe("fulfilled");
      expect(b.status).toBe("rejected");
      expect((b as PromiseRejectedResult).reason).toBeInstanceOf(EditConflictError);
      expect(versionNumbers(db, ctx.driveId, "/race.txt")).toEqual([1, 2, 3]);
      if (op.content === null) {
        await expect(cat(ctx, { path: "/race.txt" })).rejects.toThrow();
      } else {
        expect((await cat(ctx, { path: "/race.txt" })).content).toBe(op.content);
      }
    });
  }

  test("mv a b and mv b a lock their paths in the same order", async () => {
    const { ctx } = createTestContext();
    await write(ctx, { path: "/a.txt", content: "a" });
    await write(ctx, { path: "/b.txt", content: "b" });

    // Both settle (no deadlock); each moves whatever is at its source.
    const results = await Promise.allSettled([
      mv(ctx, { from: "/a.txt", to: "/b.txt" }),
      mv(ctx, { from: "/b.txt", to: "/a.txt" }),
    ]);
    expect(results[0].status).toBe("fulfilled");
  });

  test("createVersion re-checks expectedVersion inside its transaction", async () => {
    const { ctx, db } = createTestContext();
    await write(ctx, { path: "/stale.txt", content: "v1" });
    await rm(ctx, { path: "/stale.txt" });
    await write(ctx, { path: "/stale.txt", content: "v3", expectedVersion: 0 });

    // A writer whose pre-flight check saw the v2 tombstone must not commit now.
    await expect(
      createVersion(ctx, {
        path: "/stale.txt",
        s3VersionId: "",
        operation: "write",
        expectedVersion: 0,
      })
    ).rejects.toThrow(EditConflictError);
    expect(versionNumbers(db, ctx.driveId, "/stale.txt")).toEqual([1, 2, 3]);
  });
});

describe("write op — concurrent writers + UNIQUE(path, drive_id, version)", () => {
  test("two concurrent writes with the same expectedVersion: exactly one wins", async () => {
    const { ctx, db } = createTestContext();

    // Seed v1 so both racers can target expectedVersion: 1.
    await write(ctx, { path: "/race.txt", content: "v1" });

    // Issue two writes concurrently. The loser waits for the winner's path
    // lock, then fails its expectedVersion check with EditConflictError.
    const results = await Promise.allSettled([
      write(ctx, {
        path: "/race.txt",
        content: "racerA",
        expectedVersion: 1,
      }),
      write(ctx, {
        path: "/race.txt",
        content: "racerB",
        expectedVersion: 1,
      }),
    ]);

    const fulfilled = results.filter((r) => r.status === "fulfilled");
    const rejected = results.filter((r) => r.status === "rejected");

    expect(fulfilled.length).toBe(1);
    expect(rejected.length).toBe(1);

    // The loser is an EditConflictError (either from the app-layer check
    // or the DB-layer UNIQUE failure mapped to the same error type).
    const loser = rejected[0] as PromiseRejectedResult;
    expect(loser.reason).toBeInstanceOf(EditConflictError);

    // Exactly one new version row was created (v2). v1 is the seed.
    const versions = db
      .select()
      .from(schema.fileVersions)
      .where(
        and(
          eq(schema.fileVersions.path, "/race.txt"),
          eq(schema.fileVersions.driveId, ctx.driveId)
        )
      )
      .all();

    const versionNumbers = versions.map((v) => v.version).sort();
    expect(versionNumbers).toEqual([1, 2]);
  });

  test("UNIQUE(path, drive_id, version) blocks duplicate version inserts directly", async () => {
    // Sanity check on the schema constraint itself: two version rows
    // with the same (path, drive_id, version) must fail at the DB layer.
    const { ctx, db } = createTestContext();

    // First insert succeeds
    db.insert(schema.fileVersions)
      .values({
        path: "/unique.txt",
        driveId: ctx.driveId,
        version: 1,
        s3VersionId: "s3-a",
        author: ctx.userId,
        operation: "write",
        message: null,
        diffSummary: null,
        size: 0,
        etag: null,
        contentHash: null,
        createdAt: new Date(),
      })
      .run();

    // Second insert at same (path, drive_id, version) must throw.
    expect(() =>
      db
        .insert(schema.fileVersions)
        .values({
          path: "/unique.txt",
          driveId: ctx.driveId,
          version: 1,
          s3VersionId: "s3-b",
          author: ctx.userId,
          operation: "write",
          message: null,
          diffSummary: null,
          size: 0,
          etag: null,
          contentHash: null,
          createdAt: new Date(),
        })
        .run()
    ).toThrow(/UNIQUE constraint failed/);
  });
});
