import { eq, and, sql, desc } from "drizzle-orm";
import { schema, type DB } from "../db/index.js";
import type { OpContext } from "./types.js";
import { normalizePath, stripLeadingSlash } from "./paths.js";
import { EditConflictError } from "../errors.js";
import { invalidateDriveGlobListings } from "./glob-cache.js";
import { publishDriveEvent } from "../events/bus.js";

/**
 * Compute the S3 object key for a file.
 * Format: <orgId>/drives/<driveId>/<path>
 */
export function getS3Key(orgId: string, driveId: string, path: string): string {
  const normalized = stripLeadingSlash(path);
  return `${orgId}/drives/${driveId}/${normalized}`;
}

/**
 * Get the next version number for a file (monotonically increasing).
 */
export async function getNextVersion(
  ctx: OpContext,
  path: string
): Promise<number> {
  const result = ctx.db
    .select({ maxVersion: sql<number>`MAX(version)` })
    .from(schema.fileVersions)
    .where(
      and(
        eq(schema.fileVersions.path, path),
        eq(schema.fileVersions.driveId, ctx.driveId)
      )
    )
    .get();

  return (result?.maxVersion ?? 0) + 1;
}

/**
 * Look up the head version row for a (path, drive) pair. Returns the
 * version number, content hash, etag, and createdAt — the exact fields
 * the `GET /raw` HTTP handler needs to emit ETag/Last-Modified/version
 * headers without forcing the route to import drizzle directly.
 *
 * Returns `null` if no versions exist yet for that file.
 */
export interface HeadVersionRow {
  version: number;
  contentHash: string | null;
  etag: string | null;
  createdAt: Date;
}

export function getHeadVersionRow(
  ctx: Pick<OpContext, "db" | "driveId">,
  path: string
): HeadVersionRow | null {
  const row = ctx.db
    .select({
      version: schema.fileVersions.version,
      contentHash: schema.fileVersions.contentHash,
      etag: schema.fileVersions.etag,
      createdAt: schema.fileVersions.createdAt,
    })
    .from(schema.fileVersions)
    .where(
      and(
        eq(schema.fileVersions.path, path),
        eq(schema.fileVersions.driveId, ctx.driveId)
      )
    )
    .orderBy(desc(schema.fileVersions.version))
    .limit(1)
    .get();

  if (!row) return null;
  return {
    version: row.version,
    contentHash: row.contentHash ?? null,
    etag: row.etag ?? null,
    createdAt: row.createdAt,
  };
}

/** A drizzle transaction handle: the same query builder as {@link DB}. */
type Tx = Parameters<Parameters<DB["transaction"]>[0]>[0];

/** Tail of each drive path's queue of ops, keyed by `<driveId>:<path>`. */
const pathLocks = new Map<string, Promise<void>>();

/**
 * Run `fn` while this process holds the lock on every given drive path.
 *
 * Mutating ops run their `expectedVersion` check, storage write and
 * `createVersion` under it, so a second op on the same path cannot slip in
 * between the check and the commit, and a rejected writer never reaches
 * storage. Ops queue per path in arrival order. Paths are locked in sorted
 * order, so two ops that share paths (`mv a b` and `mv b a`) cannot deadlock.
 * `fn` may `unlock` a path early, once it has committed its last write there.
 */
export async function withPathLock<T>(
  ctx: Pick<OpContext, "driveId">,
  paths: string[],
  fn: (unlock: (path: string) => void) => Promise<T>
): Promise<T> {
  const releases = new Map<string, () => void>();
  try {
    for (const path of [...new Set(paths.map(normalizePath))].sort()) {
      releases.set(path, await lockPath(`${ctx.driveId}:${path}`));
    }
    return await fn((path) => {
      const normalized = normalizePath(path);
      releases.get(normalized)?.();
      releases.delete(normalized);
    });
  } finally {
    for (const release of releases.values()) release();
  }
}

async function lockPath(key: string): Promise<() => void> {
  const previous = pathLocks.get(key);
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  const tail = previous ? previous.then(() => held) : held;
  pathLocks.set(key, tail);
  await previous;
  return () => {
    release();
    if (pathLocks.get(key) === tail) pathLocks.delete(key);
  };
}

/**
 * Pre-flight optimistic-concurrency check, shared across mutating ops.
 *
 * Returns the current head version (next - 1). If `expectedVersion` is
 * passed and doesn't match the current head, throws `EditConflictError`.
 * `createVersion` repeats the check inside its transaction.
 *
 * `expectedVersion: 0` means "the file must not exist": no versions yet,
 * or the head is a delete tombstone (after `rm` or `mv`). A deleted path
 * also still matches its tombstone's version, so history-aware callers
 * keep working.
 */
export async function assertExpectedVersion(
  ctx: OpContext,
  path: string,
  expectedVersion: number | undefined
): Promise<number> {
  return checkHeadVersion(ctx.db, ctx.driveId, path, expectedVersion);
}

function checkHeadVersion(
  db: DB | Tx,
  driveId: string,
  path: string,
  expectedVersion: number | undefined
): number {
  const head = db
    .select({
      version: schema.fileVersions.version,
      operation: schema.fileVersions.operation,
    })
    .from(schema.fileVersions)
    .where(
      and(
        eq(schema.fileVersions.path, path),
        eq(schema.fileVersions.driveId, driveId)
      )
    )
    .orderBy(desc(schema.fileVersions.version))
    .limit(1)
    .get();
  const currentVersion = head?.version ?? 0;
  const matches =
    expectedVersion === currentVersion ||
    (expectedVersion === 0 && head?.operation === "delete");
  if (expectedVersion !== undefined && !matches) {
    throw new EditConflictError(
      `Expected version ${expectedVersion} but file is at version ${currentVersion}`,
      {
        path,
        suggestion: "Re-read the file to get the current version and retry",
      }
    );
  }
  return currentVersion;
}

/**
 * Return the `content_hash` of the current head version for a file, or
 * null if the file has no versions or the head row was written before
 * the column existed.
 *
 * Used by `write` (and any future op) to short-circuit identical-content
 * writes — no S3 PUT, no version row, no FTS5/embedding work.
 */
export async function getHeadContentHash(
  ctx: OpContext,
  path: string
): Promise<string | null> {
  const result = ctx.db
    .select({ contentHash: schema.fileVersions.contentHash })
    .from(schema.fileVersions)
    .where(
      and(
        eq(schema.fileVersions.path, path),
        eq(schema.fileVersions.driveId, ctx.driveId)
      )
    )
    .orderBy(desc(schema.fileVersions.version))
    .limit(1)
    .get();

  return result?.contentHash ?? null;
}

/**
 * Create a version record and update file metadata.
 *
 * The head read, the `expectedVersion` re-check, the version-row insert
 * and the files-row upsert run inside a single SQLite transaction, so a
 * writer whose pre-flight `assertExpectedVersion` went stale cannot commit,
 * and the unique-index conflict (path, drive_id, version) becomes a single
 * atomic failure surface.
 *
 * A `SQLITE_CONSTRAINT_UNIQUE` violation on `file_versions` is mapped
 * to `EditConflictError` so concurrent writers see the same conflict
 * shape that the app-layer `expectedVersion` check produces.
 */
export async function createVersion(
  ctx: OpContext,
  params: {
    path: string;
    s3VersionId: string;
    operation: "write" | "edit" | "append" | "delete" | "revert";
    message?: string;
    diffSummary?: string;
    size?: number;
    etag?: string;
    contentType?: string;
    contentHash?: string;
    /** The caller's `expectedVersion`, checked again against the head. */
    expectedVersion?: number;
  }
): Promise<number> {
  // Invalidate after storage writes but before recording the version, so failed
  // SQLite writes cannot leave a stale listing cached.
  invalidateDriveGlobListings(ctx.orgId, ctx.driveId);

  let version = 0;
  const now = new Date();

  // Wrap the version insert + files upsert in a single SQLite transaction.
  // drizzle's bun-sqlite `transaction` is sync — all queries inside are sync too.
  try {
    ctx.db.transaction((tx) => {
      version =
        checkHeadVersion(tx, ctx.driveId, params.path, params.expectedVersion) + 1;

      // Insert version record
      tx.insert(schema.fileVersions)
        .values({
          path: params.path,
          driveId: ctx.driveId,
          version,
          s3VersionId: params.s3VersionId,
          author: ctx.userId,
          operation: params.operation,
          message: params.message ?? null,
          diffSummary: params.diffSummary ?? null,
          size: params.size ?? null,
          etag: params.etag ?? null,
          contentHash: params.contentHash ?? null,
          createdAt: now,
        })
        .run();

      // Upsert file metadata
      const existing = tx
        .select()
        .from(schema.files)
        .where(
          and(
            eq(schema.files.path, params.path),
            eq(schema.files.driveId, ctx.driveId)
          )
        )
        .get();

      if (existing) {
        tx.update(schema.files)
          .set({
            size: params.size ?? existing.size,
            contentType: params.contentType ?? existing.contentType,
            author: ctx.userId,
            currentVersionId: String(version),
            modifiedAt: now,
            isDeleted: params.operation === "delete",
          })
          .where(
            and(
              eq(schema.files.path, params.path),
              eq(schema.files.driveId, ctx.driveId)
            )
          )
          .run();
      } else {
        tx.insert(schema.files)
          .values({
            path: params.path,
            driveId: ctx.driveId,
            size: params.size ?? 0,
            contentType: params.contentType ?? null,
            author: ctx.userId,
            currentVersionId: String(version),
            createdAt: now,
            modifiedAt: now,
            isDeleted: params.operation === "delete",
          })
          .run();
      }
    });
  } catch (err: any) {
    // Map SQLite UNIQUE(path, drive_id, version) failures from concurrent
    // writers to EditConflictError. Both bun:sqlite (SQLiteError) and the
    // underlying libsqlite surface the constraint name in the message.
    const msg = String(err?.message ?? "");
    if (
      msg.includes("UNIQUE constraint failed") &&
      msg.includes("file_versions")
    ) {
      throw new EditConflictError(
        `Concurrent write detected for ${params.path}: version ${version} already exists`,
        {
          path: params.path,
          suggestion:
            "Re-read the file to get the current version and retry the write",
        }
      );
    }
    throw err;
  }

  publishDriveEvent({
    type: "file.changed",
    driveId: ctx.driveId,
    path: params.path,
    version,
    operation: params.operation,
    actor: ctx.userId,
    at: now.toISOString(),
  });

  return version;
}
