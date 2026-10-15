import { useMemo, useRef, useCallback } from 'react';
import { useRecoilCallback } from 'recoil';
import { useStore, useSetAtom } from 'jotai';
import type { ResumableSteers, GenerationProtocolVersion } from '@librechat/chat';
import type { TPendingSteer, TSteerUpdatedEvent } from 'librechat-data-provider';
import type { TSubmission } from 'librechat-data-provider';
import type { QueuedMessageOrigin, PendingSteer } from '~/hooks/Chat/queue';
import {
  carriedSteerContext,
  appendAppliedSteerIds,
  collectAppliedSteerIds,
  collectDroppedSteerQuotes,
  mergeRestagedQuotes,
  insertQueuedOrigin,
  hydrateFileDeliveryMetadata,
} from '~/utils';
import {
  blockRecovery,
  canRestoreRecovery,
  recoveryDispositionsFamily,
} from '~/components/Chat/Steering/recovery';
import { queuedMessagesByConvoId } from '~/hooks/Chat/queue';
import useSteerConvert from '~/hooks/Chat/useSteerConvert';
import { liveAppliedSteerIdsAtom } from '~/store/steer';
import { useFileMapContext } from '~/Providers';
import store from '~/store';

export { ABORT_SWEEP_STATUSES } from '@librechat/chat';

/** Default sweep for a run that has genuinely ended (final/error/404): a
 *  `pending` chip behind a server id that no `on_steer_applied` will ever
 *  confirm, or a `failed` chip that never reached the server at all. Either
 *  one left behind survives past this run end and renders under whatever
 *  comes next, since `PendingSteers` reads the same atom keyed only by
 *  conversation, not by run. */
const RUN_ENDED_STATUSES: readonly PendingSteer['status'][] = ['pending', 'failed'];

/**
 * Local chips with no injection-boundary event left to resolve them.
 * `statuses` defaults to `RUN_ENDED_STATUSES` for terminals where the run is
 * actually over; pass `['failed']` for a path (like intentional abort) where
 * the run may still be live server-side and a server-ACK'd `pending` chip
 * must be left alone: the server injects it regardless, and sweeping it here
 * would resend the same words as a duplicate queued turn.
 *
 * `sending` chips are never included: they have their own in-flight POST
 * whose `onSuccess`/`onError` will settle them, and sweeping them here too
 * would race that callback, since a late ACK's re-add in
 * `resolveAcknowledgedSteer` could then double-queue the same words.
 * Uncertain failures are also excluded. Protocol v2 makes an explicit retry
 * idempotent, but a local queue conversion would create a second submission.
 */
export function selectLocalSteersForQueue(
  chips: PendingSteer[],
  statuses: readonly PendingSteer['status'][] = RUN_ENDED_STATUSES,
  /** Ids the caller has already routed elsewhere this run end; matched against
   *  both ids a chip can be known by, since either may be the one excluded. */
  excluded: ReadonlySet<string> = new Set(),
): TPendingSteer[] {
  const allowed = new Set(statuses);
  return chips
    .filter(
      (steer) =>
        allowed.has(steer.status) &&
        steer.deliveryUncertain !== true &&
        !excluded.has(steer.steerId) &&
        (steer.clientSteerId == null || !excluded.has(steer.clientSteerId)),
    )
    .map((steer) => ({
      steerId: steer.steerId,
      ...(steer.clientSteerId && { clientSteerId: steer.clientSteerId }),
      text: steer.text,
      createdAt: steer.createdAt,
      ...(steer.files && steer.files.length > 0 && { files: steer.files }),
      ...(steer.queuedOrigin && { queuedOrigin: steer.queuedOrigin }),
    }));
}

/**
 * The pane's steer chips as a resumable run drives them: settling chips the stream reports
 * applied, re-seeding them from a resume or status snapshot, and turning steers the run never
 * injected into queued follow-ups.
 */
export default function useSteerChips(): ResumableSteers {
  const jotaiStore = useStore();
  const fileMap = useFileMapContext();
  const fileMapRef = useRef(fileMap);
  fileMapRef.current = fileMap;
  const setLiveAppliedSteerIds = useSetAtom(liveAppliedSteerIdsAtom);

  const restoreQueuedSubmission = useCallback(
    (failedSubmission: TSubmission, expectedPredecessorCreatedAt?: number) => {
      const conversationId = failedSubmission.conversation?.conversationId;
      const origin = failedSubmission.queuedMessageOrigin as QueuedMessageOrigin | undefined;
      if (!conversationId || origin == null) {
        return;
      }
      jotaiStore.set(queuedMessagesByConvoId(conversationId), (prev) =>
        canRestoreRecovery(jotaiStore.get(recoveryDispositionsFamily(conversationId)), origin.item)
          ? insertQueuedOrigin(prev, origin, expectedPredecessorCreatedAt)
          : prev,
      );
    },
    [jotaiStore],
  );

  /** Removes the pending chip once its steer is injected (the inline content
   *  part becomes the durable record), and records the id so a 202 ACK that
   *  arrives AFTER the applied event drops its chip instead of re-minting it. */
  const resolveSteerChip = useRecoilCallback(
    ({ snapshot, set }) =>
      (
        conversationId: string,
        steerId: string,
        clientSteerId?: string,
        appliedPartQuotes?: string[],
      ): string[] => {
        const settledIds = clientSteerId ? [steerId, clientSteerId] : [steerId];
        /** A part applied by a pre-quotes server carries no quotes while the
         *  chip being settled may hold the only copy of the user's excerpts
         *  (its 202 was lost, so the ACK-echo restore never ran). Re-stage
         *  them before removal; `mergeRestagedQuotes` keeps this idempotent
         *  with the ACK path for the same excerpts. */
        if (appliedPartQuotes == null || appliedPartQuotes.length === 0) {
          const chips = snapshot
            .getLoadable(store.pendingSteersByConvoId(conversationId))
            .getValue();
          const droppedQuotes = chips.find(
            (steer) => settledIds.includes(steer.steerId) && (steer.quotes?.length ?? 0) > 0,
          )?.quotes;
          if (droppedQuotes != null && droppedQuotes.length > 0) {
            set(store.pendingQuotesByConvoId(conversationId), (prev) =>
              mergeRestagedQuotes(prev, droppedQuotes),
            );
          }
        }
        /** Ids seen by the durable set for the first time; the caller stamps
         *  these as live-applied only when it actually commits the inline
         *  part, so an abandoned placement (navigation away, placeholder never
         *  found) or a replayed event cannot arm a draw-in with no part
         *  mounting to consume it. */
        const alreadyApplied = snapshot
          .getLoadable(store.appliedSteerIdsByConvoId(conversationId))
          .getValue();
        const firstApplication = settledIds.filter((id) => !alreadyApplied.includes(id));
        set(store.appliedSteerIdsByConvoId(conversationId), (prev) =>
          appendAppliedSteerIds(prev, settledIds),
        );
        set(store.pendingSteersByConvoId(conversationId), (prev) =>
          prev.some((steer) => settledIds.includes(steer.steerId))
            ? prev.filter((steer) => !settledIds.includes(steer.steerId))
            : prev,
        );
        /** A terminal boundary may have re-homed this steer while its applied
         *  event was waiting for the response placeholder. The applied event
         *  is authoritative, so evict that recovery copy before it can be
         *  drained as a duplicate follow-up. */
        jotaiStore.set(queuedMessagesByConvoId(conversationId), (prev) =>
          prev.some(
            (item) =>
              (item.recoverySteerId != null && settledIds.includes(item.recoverySteerId)) ||
              (item.recoveryClientSteerId != null &&
                settledIds.includes(item.recoveryClientSteerId)),
          )
            ? prev.filter(
                (item) =>
                  !(
                    (item.recoverySteerId != null && settledIds.includes(item.recoverySteerId)) ||
                    (item.recoveryClientSteerId != null &&
                      settledIds.includes(item.recoveryClientSteerId))
                  ),
              )
            : prev,
        );
        return firstApplication;
      },
    [],
  );

  /** Capability can change while the same generation resumes after HITL.
   * Patch labels in place; this event never removes/reorders a chip. */
  const updateSteerChips = useRecoilCallback(
    ({ set }) =>
      (event: TSteerUpdatedEvent) => {
        const byId = new Map<string, TSteerUpdatedEvent['steers'][number]>();
        for (const steer of event.steers) {
          byId.set(steer.steerId, steer);
          if (steer.clientSteerId) {
            byId.set(steer.clientSteerId, steer);
          }
        }
        const acceptedClientIds = event.steers.flatMap((steer) =>
          steer.clientSteerId ? [steer.clientSteerId] : [],
        );
        if (acceptedClientIds.length > 0) {
          set(store.acceptedSteerClientIdsByConvoId(event.conversationId), (prev) =>
            appendAppliedSteerIds(prev, acceptedClientIds),
          );
        }
        set(store.pendingSteersByConvoId(event.conversationId), (prev) => {
          let changed = false;
          const next = prev.map((steer) => {
            const update = byId.get(steer.steerId);
            if (update == null) {
              return steer;
            }
            const correlated =
              update.clientSteerId === steer.steerId && steer.steerId !== update.steerId;
            const base = correlated
              ? {
                  ...steer,
                  steerId: update.steerId,
                  clientSteerId: update.clientSteerId,
                  status: 'pending' as const,
                }
              : steer;
            if (update.preemptRevision < (base.preemptRevision ?? 0)) {
              changed ||= base !== steer;
              return base;
            }
            if (
              Boolean(base.preempt) === update.preempt &&
              base.preemptRevision === update.preemptRevision
            ) {
              changed ||= base !== steer;
              return base;
            }
            changed = true;
            if (update.preempt) {
              return {
                ...base,
                preempt: true,
                preemptRevision: update.preemptRevision,
              };
            }
            const { preempt: _preempt, ...ordinary } = base;
            return { ...ordinary, preemptRevision: update.preemptRevision };
          });
          return changed ? next : prev;
        });
      },
    [],
  );

  /** Replaces the chip list with the server's still-queued steers (reconnect).
   *  Local `failed` entries are kept so their text stays recoverable, and a
   *  reseeded chip keeps its quotes/skill picks (from the local chip, or the
   *  server item's persisted quotes when no chip survives). */
  const seedSteerChips = useRecoilCallback(
    ({ set }) =>
      (
        conversationId: string,
        steers: TPendingSteer[],
        generationCreatedAt?: number,
        generationProtocolVersion: GenerationProtocolVersion = 1,
      ) => {
        const acceptedClientIds = steers.flatMap((steer) =>
          steer.clientSteerId ? [steer.clientSteerId] : [],
        );
        if (acceptedClientIds.length > 0) {
          set(store.acceptedSteerClientIdsByConvoId(conversationId), (prev) =>
            appendAppliedSteerIds(prev, acceptedClientIds),
          );
        }
        set(store.pendingSteersByConvoId(conversationId), (prev) => {
          const chipById = new Map(prev.map((chip) => [chip.steerId, chip]));
          const claimedIds = new Set(
            steers.flatMap((steer) =>
              steer.clientSteerId ? [steer.steerId, steer.clientSteerId] : [steer.steerId],
            ),
          );
          return [
            ...steers.map((steer) => {
              const localChip =
                chipById.get(steer.steerId) ??
                (steer.clientSteerId ? chipById.get(steer.clientSteerId) : undefined);
              const keepLocalPreempt =
                (localChip?.preemptRevision ?? 0) > (steer.preemptRevision ?? 0);
              const chipGenerationCreatedAt = generationCreatedAt ?? localChip?.generationCreatedAt;
              const restoredFiles = hydrateFileDeliveryMetadata(
                steer.files,
                localChip?.files,
                fileMapRef.current,
              );
              return {
                steerId: steer.steerId,
                ...(steer.clientSteerId && { clientSteerId: steer.clientSteerId }),
                text: steer.text,
                status: 'pending' as const,
                createdAt: steer.createdAt ?? Date.now(),
                ...(restoredFiles && restoredFiles.length > 0 && { files: restoredFiles }),
                ...((keepLocalPreempt ? localChip?.preempt : steer.preempt) === true && {
                  preempt: true,
                }),
                ...((keepLocalPreempt ? localChip?.preemptRevision : steer.preemptRevision) !=
                  null && {
                  preemptRevision: keepLocalPreempt
                    ? localChip?.preemptRevision
                    : steer.preemptRevision,
                }),
                ...(localChip?.queuedOrigin && { queuedOrigin: localChip.queuedOrigin }),
                ...(chipGenerationCreatedAt != null && {
                  generationCreatedAt: chipGenerationCreatedAt,
                }),
                generationProtocolVersion,
                // The local chip carries skill picks the server never sees; a
                // fresh tab has no chip, so fall back to the server item's
                // persisted quotes rather than reseeding the chip without them.
                ...carriedSteerContext(localChip ?? steer),
              };
            }),
            ...prev.filter((steer) => steer.status === 'failed' && !claimedIds.has(steer.steerId)),
          ];
        });
      },
    [],
  );

  const settleAppliedSteerParts = useRecoilCallback(
    ({ snapshot, set }) =>
      (conversationId: string, values: unknown[] | undefined) => {
        const ids = collectAppliedSteerIds(values);
        if (ids.length === 0) {
          return;
        }
        const settled = new Set(ids);
        /** Chips settled by quote-less applied parts hold the only copy of
         *  their excerpts (a pre-quotes server injected the words bare), so
         *  re-stage them as composer chips before the removal below. */
        const droppedQuotes = collectDroppedSteerQuotes(
          values,
          snapshot.getLoadable(store.pendingSteersByConvoId(conversationId)).getValue(),
        );
        if (droppedQuotes.length > 0) {
          set(store.pendingQuotesByConvoId(conversationId), (prev) =>
            mergeRestagedQuotes(prev, droppedQuotes),
          );
        }
        set(store.appliedSteerIdsByConvoId(conversationId), (prev) =>
          appendAppliedSteerIds(prev, ids),
        );
        set(store.pendingSteersByConvoId(conversationId), (prev) =>
          prev.filter((steer) => !settled.has(steer.steerId)),
        );
      },
    [],
  );

  /** Converts steers that never reached an injection boundary into queued
   *  follow-up messages (reported on the run's final/abort event; the abort
   *  HTTP response consumes the same data as a fallback in useChatHelpers). */
  const convertSteersToQueued = useSteerConvert();

  /** Sweeps this conversation's local chips (see `selectLocalSteersForQueue`)
   *  into queued follow-ups. Error events carry no `pendingSteers` payload of
   *  their own (the server drops its copy on failure); this is the only
   *  source of truth for those runs; the `final` and error/404 terminals call
   *  it (default `statuses`, i.e. `pending || failed`) alongside their own
   *  server-reported list as a backstop for `failed` chips, which never rode
   *  that list at all. The intentional-abort path passes `statuses: ['failed']`
   *  since the run may still be live server-side there (see its call site). */
  const convertLocalSteersToQueued = useRecoilCallback(
    ({ snapshot }) =>
      (
        conversationId: string,
        options?: {
          claimParked?: boolean;
          statuses?: readonly PendingSteer['status'][];
          excludeSteerIds?: Iterable<string>;
          generationProtocolVersion?: GenerationProtocolVersion;
        },
      ) => {
        const chips = snapshot.getLoadable(store.pendingSteersByConvoId(conversationId)).getValue();
        const generationProtocolVersion =
          options?.generationProtocolVersion ??
          snapshot
            .getLoadable(store.activeGenerationProtocolVersionByConvoId(conversationId))
            .getValue();
        const activeStatuses = options?.statuses ?? RUN_ENDED_STATUSES;
        const excluded = new Set(options?.excludeSteerIds ?? []);
        const accepted = selectLocalSteersForQueue(
          chips,
          activeStatuses.filter((status) => status === 'pending'),
          excluded,
        );
        if (accepted.length > 0) {
          convertSteersToQueued(conversationId, accepted, {
            claimParked: options?.claimParked,
            generationProtocolVersion,
          });
        }
        const failed = selectLocalSteersForQueue(
          chips,
          activeStatuses.filter((status) => status === 'failed'),
          excluded,
        );
        if (failed.length > 0) {
          convertSteersToQueued(conversationId, failed, {
            generationProtocolVersion,
            bindRecoverySource: false,
            /** Swept out of the chip surface so it cannot render under the next
             *  reply, but the server refused these words: the rail holds the row
             *  for Retry or an explicit send instead of draining it automatically. */
            needsExplicitSend: true,
          });
        }
      },
    [convertSteersToQueued],
  );

  return useMemo<ResumableSteers>(
    () => ({
      resolve: resolveSteerChip,
      isRecoveryHeld: (conversationId, recoverySteerId) =>
        jotaiStore.get(recoveryDispositionsFamily(conversationId))[recoverySteerId] != null,
      blockRecovery: (conversationId, recoverySteerId) =>
        jotaiStore.set(recoveryDispositionsFamily(conversationId), (previous) =>
          blockRecovery(previous, recoverySteerId),
        ),
      restoreQueued: restoreQueuedSubmission,
      update: updateSteerChips,
      markLiveApplied: (steerId) =>
        setLiveAppliedSteerIds((prev) => appendAppliedSteerIds(prev, [steerId])),
      seed: seedSteerChips,
      appliedIds: collectAppliedSteerIds,
      settleApplied: settleAppliedSteerParts,
      convertToQueued: convertSteersToQueued,
      convertLocalToQueued: convertLocalSteersToQueued,
    }),
    [
      jotaiStore,
      resolveSteerChip,
      restoreQueuedSubmission,
      updateSteerChips,
      setLiveAppliedSteerIds,
      seedSteerChips,
      settleAppliedSteerParts,
      convertSteersToQueued,
      convertLocalSteersToQueued,
    ],
  );
}
