import { useMemo, useEffect, useState, useRef, useCallback } from 'react';
import { v4 } from 'uuid';
import { useStore, useSetAtom } from 'jotai';
import { useQueryClient } from '@tanstack/react-query';
import { useSetRecoilState, useRecoilCallback } from 'recoil';
import { Constants, QueryKeys, dataService } from 'librechat-data-provider';
import {
  getOptimisticMessages,
  isInitialNewConversation,
  getStreamStartFailureText,
  hydrateSubmissionMessages,
  buildOptimisticConversation,
  ResumableRun,
  documentVisibility,
  attachResumableStream,
  mountSubmission,
  releaseSubmission,
} from '@librechat/chat';
import type {
  TMessage,
  TSubmission,
  TConversation,
  TPendingSteer,
  EventSubmission,
  TSteerUpdatedEvent,
} from 'librechat-data-provider';
import type { ResumableHost, ResumableCache, GenerationProtocolVersion } from '@librechat/chat';
import type { QueuedMessageOrigin, DrainAfterAbort, RunEnd } from '~/hooks/Chat/queue';
import type { EventHandlerParams } from './useEventHandlers';
import type { ActiveJobsResponse } from '~/data-provider';
import type { TResData, TFinalResData } from '~/common';
import type { PendingSteer } from '~/hooks/Chat/queue';
import {
  logger,
  clearComposerDrafts,
  applySteerPart,
  carriedSteerContext,
  resolveRunEndTarget,
  findSteerMessageIndex,
  applyActivityLabelPart,
  applyReasoningLabel,
  offsetActivityPhaseBoundary,
  findActivityLabelMessageIndex,
  findReasoningLabelMessageIndex,
  appendAppliedSteerIds,
  collectAppliedSteerIds,
  collectDroppedSteerQuotes,
  mergeRestagedQuotes,
  removeConvoFromAllQueries,
  upsertConvoInAllQueries,
  invalidateConversationLists,
  countTrailingOutputChars,
  markStreamStartFailedMetadata,
  insertQueuedOrigin,
  hydrateFileDeliveryMetadata,
} from '~/utils';
import {
  resolveDetachedRunEnd,
  isDetachedRunManual,
  stopRequestedByConvoId,
  detachedRunByConvoId,
  isStoppedRun,
  drainAfterAbortByIndex,
  queuedMessagesByConvoId,
  runEndByIndex,
} from '~/hooks/Chat/queue';
import {
  useGetUserBalance,
  fetchStreamStatus,
  useGetStartupConfig,
  queueTitleGeneration,
  streamStatusQueryKey,
  fetchConversationMessages,
} from '~/data-provider';
import {
  recoveryDispositionsFamily,
  canRestoreRecovery,
  blockRecovery,
} from '~/components/Chat/Steering/recovery';
import { pendingApprovalActionFamily } from '~/components/Chat/approval/state';
import { abortScrollFamily, showStopButtonByIndex } from '~/store/generation';
import { useChatTransport } from '~/Providers/ChatTransportContext';
import useSteerConvert from '~/hooks/Chat/useSteerConvert';
import { liveAppliedSteerIdsAtom } from '~/store/steer';
import { reopenTermsOnRefusal } from '~/utils/terms';
import { useAuthContext } from '~/hooks/AuthContext';
import { fetchConvoSnapshot } from '~/utils/convos';
import useEventHandlers from './useEventHandlers';
import { useFileMapContext } from '~/Providers';
import useUsageHandler from './useUsageHandler';
import useLocalize from '~/hooks/useLocalize';
import store from '~/store';

type ChatHelpers = Pick<
  EventHandlerParams,
  'setMessages' | 'getMessages' | 'setConversation' | 'setIsSubmitting' | 'newConversation'
>;

/**
 * Reconciliation often crosses async query boundaries. By the time an old
 * generation settles, this pane may have armed "interrupt & send" for a new
 * conversation/generation. Recoil applies functional setters against the
 * latest value, so only consume the exact intent owned by this terminal path.
 */
const clearMatchingDrainAfterAbort = (
  armed: DrainAfterAbort | false,
  conversationId: string,
  generationCreatedAt: number,
): DrainAfterAbort | false =>
  armed !== false &&
  armed.conversationId === conversationId &&
  armed.generationCreatedAt === generationCreatedAt
    ? false
    : armed;

const getStreamStartFailureData = (errorData?: unknown, termsRefusalText?: string): TResData =>
  ({
    text: termsRefusalText ?? getStreamStartFailureText(errorData),
    metadata: markStreamStartFailedMetadata(),
  }) as unknown as TResData;

const replaceNewConversationUrl = (conversationId: string) => {
  if (window.location.pathname !== `/c/${Constants.NEW_CONVO}`) {
    return;
  }

  window.history.replaceState(
    window.history.state,
    '',
    `/c/${conversationId}${window.location.search}`,
  );
};

/** Default sweep for a run that has genuinely ended (final/error/404): a
 *  `pending` chip behind a server id that no `on_steer_applied` will ever
 *  confirm, or a `failed` chip that never reached the server at all. Either
 *  one left behind survives past this run end and renders under whatever
 *  comes next, since `PendingSteers` reads the same atom keyed only by
 *  conversation, not by run. */
const RUN_ENDED_STATUSES: readonly PendingSteer['status'][] = ['pending', 'failed'];

/** Sweep for an intentional abort, where the run may still be live
 *  server-side: a server-ACK'd `pending` chip is injected regardless, so
 *  sweeping it here would send the same words a second time as a queued turn. */
export const ABORT_SWEEP_STATUSES: readonly PendingSteer['status'][] = ['failed'];

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
 * Hook for resumable SSE streams.
 * Separates generation start (POST) from stream subscription (GET EventSource).
 * Supports auto-reconnection with exponential backoff.
 *
 * Key behavior:
 * - Navigation away does NOT abort the generation (just closes SSE)
 * - Only explicit abort (via stop button → backend abort endpoint) stops generation
 * - Backend emits `done` event with `aborted: true` on abort, handled via finalHandler
 */
export default function useResumableSSE(
  submission: TSubmission | null,
  chatHelpers: ChatHelpers,
  isAddedRequest = false,
  runIndex = 0,
) {
  const [run] = useState(() => new ResumableRun());
  const jotaiStore = useStore();
  const localize = useLocalize();
  const queryClient = useQueryClient();
  const setActiveRunId = useSetRecoilState(store.activeRunFamily(runIndex));

  const { token, isAuthenticated } = useAuthContext();
  const transport = useChatTransport();
  const fileMap = useFileMapContext();
  const fileMapRef = useRef(fileMap);
  fileMapRef.current = fileMap;
  const { setMessages, getMessages, setConversation, setIsSubmitting, newConversation } =
    chatHelpers;

  /**
   * Optimistically add a job ID to the active jobs cache.
   * Called when generation starts.
   */
  const addActiveJob = useCallback(
    (jobId: string) => {
      queryClient.setQueryData<ActiveJobsResponse>([QueryKeys.activeJobs], (old) => ({
        activeJobIds: [...new Set([...(old?.activeJobIds ?? []), jobId])],
      }));
    },
    [queryClient],
  );

  /**
   * Optimistically remove a job ID from the active jobs cache.
   * Called when generation completes, aborts, or errors.
   */
  const removeActiveJob = useCallback(
    (jobId: string) => {
      queryClient.setQueryData<ActiveJobsResponse>([QueryKeys.activeJobs], (old) => ({
        activeJobIds: (old?.activeJobIds ?? []).filter((id) => id !== jobId),
      }));
    },
    [queryClient],
  );

  const addOptimisticConversation = useCallback(
    (conversationId: string, currentSubmission: TSubmission): TSubmission => {
      if (!isInitialNewConversation(currentSubmission)) {
        return currentSubmission;
      }

      const optimisticConversation = buildOptimisticConversation(currentSubmission, conversationId);
      const optimisticMessages = getOptimisticMessages(
        currentSubmission,
        conversationId,
        getMessages(),
      );

      queryClient.setQueryData<TConversation>(
        [QueryKeys.conversation, conversationId],
        (current) => current ?? optimisticConversation,
      );
      queryClient.setQueryData<TMessage[]>(
        [QueryKeys.messages, conversationId],
        optimisticMessages,
      );
      queryClient.setQueryData<TMessage[]>(
        [QueryKeys.messages, Constants.NEW_CONVO],
        optimisticMessages,
      );
      upsertConvoInAllQueries(queryClient, optimisticConversation);

      return hydrateSubmissionMessages(currentSubmission, conversationId);
    },
    [getMessages, queryClient],
  );

  const resumableCache = useMemo<ResumableCache>(
    () => ({
      invalidateMessages: (conversationId, refetch) =>
        queryClient.invalidateQueries({
          queryKey: [QueryKeys.messages, conversationId],
          refetchType: refetch,
        }),
      fetchMessages: (conversationId, fallback) =>
        queryClient.fetchQuery<TMessage[]>({
          queryKey: [QueryKeys.messages, conversationId],
          ...(fallback === true && {
            // The first stream can finish before CREATED mounts the saved-chat query.
            queryFn: () => fetchConversationMessages(conversationId),
          }),
        }),
      refreshConversation: (conversationId) =>
        queryClient.invalidateQueries({
          queryKey: [QueryKeys.conversation, conversationId],
          refetchType: 'all',
        }),
      refreshConversationLists: () => invalidateConversationLists(queryClient),
      refreshPinnedConversations: () =>
        queryClient.invalidateQueries({ queryKey: [QueryKeys.pinnedConversations] }),
      setStreamStatus: (conversationId, status) =>
        queryClient.setQueryData(streamStatusQueryKey(conversationId), status),
      removeStreamStatus: (conversationId) =>
        queryClient.removeQueries({ queryKey: streamStatusQueryKey(conversationId) }),
      invalidateStreamStatus: (conversationId) =>
        queryClient.invalidateQueries({ queryKey: streamStatusQueryKey(conversationId) }),
      addActiveJob,
      removeActiveJob,
      fetchConversation: (conversationId) =>
        fetchConvoSnapshot(queryClient, conversationId, () =>
          dataService.getConversationById(conversationId),
        ),
      setConversation: (conversationId, conversation) =>
        queryClient.setQueryData([QueryKeys.conversation, conversationId], conversation),
      upsertConversation: (conversation) =>
        upsertConvoInAllQueries(queryClient, conversation, true, 'snapshot'),
      removeConversation: (conversationId) =>
        removeConvoFromAllQueries(queryClient, conversationId),
      getConversation: (conversationId) =>
        queryClient.getQueryData<TConversation>([QueryKeys.conversation, conversationId]),
      removeConversationQueries: (conversationId) => {
        queryClient.removeQueries({ queryKey: [QueryKeys.conversation, conversationId] });
        queryClient.removeQueries({ queryKey: [QueryKeys.messages, conversationId] });
      },
      queueTitleGeneration,
      addOptimisticConversation,
    }),
    [addActiveJob, addOptimisticConversation, queryClient, removeActiveJob],
  );
  const [_completed, setCompleted] = useState(new Set());
  const [streamId, setStreamId] = useState<string | null>(null);
  const setAbortScroll = useSetAtom(abortScrollFamily(runIndex));
  const setSubmission = useSetRecoilState(store.submissionByIndex(runIndex));
  const setShowStopButton = useSetAtom(showStopButtonByIndex(runIndex));
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
        /** Ids seen by the durable set for the first time — the caller stamps
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
                createdAt: steer.createdAt ?? run.scheduler.now(),
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
         *  their excerpts (a pre-quotes server injected the words bare) —
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

  /** Conversation ids are reused by successive agent turns. Keep the exact
   * live epoch beside the UI controls, and only clear it if the terminal event
   * still belongs to the epoch that installed it. */
  const updateActiveGenerationCreatedAt = useRecoilCallback(
    ({ snapshot, set }) =>
      (
        conversationId: string,
        next: number | null,
        expected?: number,
        generationProtocolVersion: GenerationProtocolVersion = 1,
      ): boolean => {
        const state = store.activeGenerationCreatedAtByConvoId(conversationId);
        if (expected != null && snapshot.getLoadable(state).getValue() !== expected) {
          return false;
        }
        set(state, next);
        set(
          store.activeGenerationProtocolVersionByConvoId(conversationId),
          next == null ? 1 : generationProtocolVersion,
        );
        return true;
      },
    [],
  );

  const publishRunEnd = useSetAtom(runEndByIndex(runIndex));
  const setRunEnd = useCallback(
    (end: RunEnd, options?: { retainDetached?: boolean }) => {
      run.runEndPublished = true;
      /** The run's end reached the drain here, so a marker this pane left for the same run is
       *  settled and must not be resolved a second time on a later visit. A provisional end that
       *  cannot drain the queue keeps the marker, so a later read of history can still settle it. */
      if (end.conversationId != null && options?.retainDetached !== true) {
        const detachedFamily = detachedRunByConvoId(end.conversationId);
        const detached = jotaiStore.get(detachedFamily);
        if (
          detached != null &&
          (end.generationCreatedAt == null ||
            detached.generationCreatedAt === end.generationCreatedAt)
        ) {
          jotaiStore.set(detachedFamily, null);
        }
      }
      publishRunEnd(end);
    },
    [jotaiStore, publishRunEnd, run],
  );
  const setDrainAfterAbort = useSetAtom(drainAfterAbortByIndex(runIndex));
  const clearDrainAfterAbort = useCallback(
    (conversationId: string, generationCreatedAt?: number) => {
      if (generationCreatedAt == null) {
        return;
      }
      setDrainAfterAbort((armed) =>
        clearMatchingDrainAfterAbort(armed, conversationId, generationCreatedAt),
      );
    },
    [setDrainAfterAbort],
  );

  const {
    stepHandler: rawStepHandler,
    finalHandler,
    errorHandler,
    clearStepMaps,
    messageHandler,
    contentHandler,
    createdHandler,
    titleHandler,
    syncStepMessage,
    prunePtcTraces,
    attachmentHandler,
    resetContentHandler,
    flushPendingDeltas,
  } = useEventHandlers({
    setMessages,
    getMessages,
    setCompleted,
    isAddedRequest,
    runIndex,
    setConversation,
    setIsSubmitting,
    newConversation,
    setShowStopButton,
  });

  /**
   * Run steps and activity labels must resolve indices in ONE space, and the
   * resume SYNC boundary that invalidates the edit prefix is owned here, so
   * this transport stamps its cleared state onto every dispatched
   * submission. `useStepHandler` reads the flag (alongside the captured
   * `editPrefixLength`) instead of measuring the live
   * `initialResponse.content`, which SYNC replaces with the server's
   * completion-local snapshot.
   *
   * Only allocates once the prefix is actually cleared; before that, and on
   * the non-resumable transport, the submission passes through untouched.
   */
  const stepHandler = useCallback(
    (...[event, submission]: Parameters<typeof rawStepHandler>) => {
      const eventSubmission = run.editPrefixCleared
        ? ({ ...submission, editPrefixCleared: true } as EventSubmission)
        : submission;
      rawStepHandler(event, eventSubmission);
      if (eventSubmission.editPrefixFirstPartFolded === true) {
        run.editPrefixFirstPartFolded = true;
      }
    },
    [rawStepHandler, run],
  );

  const { data: startupConfig } = useGetStartupConfig();
  const balanceQuery = useGetUserBalance({
    enabled: !!isAuthenticated && startupConfig?.balance?.enabled,
  });
  const {
    bindResponse,
    contextHandler,
    usageHandler,
    tapStream,
    tapContent,
    finalizeUsage,
    backfillUsage,
    resetLive,
    seedLive,
  } = useUsageHandler();

  /**
   * Subscribe to stream via SSE library (supports custom headers)
   * Follows same auth pattern as useSSE
   * @param isResume - If true, adds ?resume=true to trigger sync event from server
   */
  const buildHost = useCallback(
    (): ResumableHost => ({
      logger,
      visibility: documentVisibility,
      fetchStatus: fetchStreamStatus,
      subscribe: (next) => attachResumableStream(run, buildHost(), next),
      cache: resumableCache,
      queue: {
        clearSupersededMarkers: (conversationId, isSameGeneration) => {
          const stopFamily = stopRequestedByConvoId(conversationId);
          const stopped = jotaiStore.get(stopFamily);
          if (stopped != null && !isSameGeneration(stopped.generationCreatedAt)) {
            jotaiStore.set(stopFamily, null);
          }
          const detachedFamily = detachedRunByConvoId(conversationId);
          const detached = jotaiStore.get(detachedFamily);
          if (detached != null && !isSameGeneration(detached.generationCreatedAt)) {
            jotaiStore.set(detachedFamily, null);
          }
        },
        isStopped: (conversationId, generationCreatedAt) =>
          isStoppedRun(jotaiStore.get(stopRequestedByConvoId(conversationId)), generationCreatedAt),
        rememberDetachedRun: (conversationId, detachedRun) =>
          jotaiStore.set(detachedRunByConvoId(conversationId), detachedRun),
      },
      pane: {
        setIsSubmitting,
        setShowStopButton,
        setSubmission,
        setStreamId,
        setGenerationEpoch: updateActiveGenerationCreatedAt,
        setPendingAction: (conversationId, action) =>
          jotaiStore.set(pendingApprovalActionFamily(conversationId), action),
        clearComposerDrafts: (conversationId, options) =>
          clearComposerDrafts(runIndex, conversationId, options),
        publishRunEnd: setRunEnd,
        setAbortScroll,
        setConversation,
        startNewConversation: newConversation
          ? () => newConversation({ template: { conversationId: String(Constants.NEW_CONVO) } })
          : undefined,
        replaceNewConversationUrl,
        renewActiveRun: () => setActiveRunId(v4()),
        runEndTarget: resolveRunEndTarget,
        clearDrainAfterAbort,
        refreshBalance: () => {
          (startupConfig?.balance?.enabled ?? false) && balanceQuery.refetch();
        },
        updateConversation: isAddedRequest ? undefined : setConversation,
        resolveDetachedEnd: (conversationId, messages) => {
          const detachedRun = jotaiStore.get(detachedRunByConvoId(conversationId));
          const resolvable =
            detachedRun != null &&
            !isDetachedRunManual(jotaiStore.get, conversationId, detachedRun);
          return {
            resolvable,
            end:
              resolvable && messages != null
                ? resolveDetachedRunEnd(conversationId, detachedRun, messages)
                : null,
          };
        },
      },
      steers: {
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
      },
      events: {
        stepHandler,
        showStartFailure: (failedSubmission, errorData, reason) => {
          let text: string | undefined;
          if (reason === 'recovery_held') {
            text = localize('com_ui_steer_recovery_held');
          } else if (reopenTermsOnRefusal(queryClient, errorData)) {
            text = localize('com_error_terms_not_accepted');
          }
          errorHandler({
            data: getStreamStartFailureData(errorData, text),
            submission: failedSubmission as EventSubmission,
          });
        },
        finalHandler: (data, finalSubmission) =>
          finalHandler(data as TFinalResData, finalSubmission),
        createdHandler,
        attachmentHandler,
        titleHandler,
        contentHandler,
        messageHandler,
        prunePtcTraces,
        resetContentHandler,
        syncStepMessage,
        clearStepMaps,
        flushPendingDeltas,
        resetLive,
        showError: (text, errorSubmission) =>
          errorHandler({
            data: { text } as unknown as Parameters<typeof errorHandler>[0]['data'],
            submission: errorSubmission as EventSubmission,
          }),
      },
      messages: { get: getMessages, set: setMessages },
      usage: {
        bindResponse,
        contextHandler,
        usageHandler,
        tapStream,
        tapContent,
        finalizeUsage,
        backfillUsage,
        seedLiveFromContent: (content, usageSubmission) =>
          seedLive(countTrailingOutputChars(content), usageSubmission),
      },
      stream: transport.stream({ token }),
      start: transport.start,
      exposeDebugControls: import.meta.env.DEV
        ? ({ killNetwork, closeClean }) => {
            // Debug hooks for testing reconnection vs clean close behavior (dev only)
            const debugWindow = window as Window & {
              __killNetwork?: () => void;
              __closeClean?: () => void;
            };
            debugWindow.__killNetwork = killNetwork;
            debugWindow.__closeClean = closeClean;
          }
        : undefined,
      parts: {
        applySteer: applySteerPart,
        findSteerTarget: findSteerMessageIndex,
        applyActivityLabel: applyActivityLabelPart,
        findActivityLabelTarget: findActivityLabelMessageIndex,
        offsetActivityPhaseBoundary,
        applyReasoningLabel,
        findReasoningLabelTarget: findReasoningLabelMessageIndex,
      },
    }),
    [
      run,
      newConversation,
      localize,
      queryClient,
      restoreQueuedSubmission,
      resumableCache,
      isAddedRequest,
      setConversation,
      runIndex,
      token,
      transport,
      setAbortScroll,
      setActiveRunId,
      setShowStopButton,
      finalHandler,
      createdHandler,
      attachmentHandler,
      titleHandler,
      stepHandler,
      contentHandler,
      resetContentHandler,
      syncStepMessage,
      prunePtcTraces,
      clearStepMaps,
      messageHandler,
      errorHandler,
      flushPendingDeltas,
      setIsSubmitting,
      getMessages,
      setMessages,
      startupConfig?.balance?.enabled,
      balanceQuery,
      bindResponse,
      contextHandler,
      usageHandler,
      tapStream,
      tapContent,
      finalizeUsage,
      backfillUsage,
      resetLive,
      seedLive,
      setRunEnd,
      clearDrainAfterAbort,
      resolveSteerChip,
      setLiveAppliedSteerIds,
      updateSteerChips,
      seedSteerChips,
      settleAppliedSteerParts,
      convertSteersToQueued,
      convertLocalSteersToQueued,
      setSubmission,
      updateActiveGenerationCreatedAt,
      jotaiStore,
    ],
  );

  useEffect(() => {
    const host = buildHost();
    if (!submission || Object.keys(submission).length === 0) {
      releaseSubmission(run, host);
      return;
    }
    return mountSubmission(run, host, submission);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [submission]);

  return { streamId };
}
