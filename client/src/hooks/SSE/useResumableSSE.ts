import { useMemo, useEffect, useState, useRef, useCallback } from 'react';
import { v4 } from 'uuid';
import { useStore, useSetAtom } from 'jotai';
import { useQueryClient } from '@tanstack/react-query';
import { useSetRecoilState, useRecoilCallback } from 'recoil';
import {
  Constants,
  QueryKeys,
  dataService,
  createPayload,
  removeNullishValues,
} from 'librechat-data-provider';
import {
  toStartGenerationError,
  waitForRetryDelay,
  getRecoverySteerId,
  getRetryAfterDelay,
  isServerNotReadyError,
  getOptimisticMessages,
  isResumedStartResponse,
  isSettledStartResponse,
  isRetryableNetworkError,
  isReplacedStartResponse,
  isInitialNewConversation,
  getStreamStartFailureText,
  keepLocalCodeApprovalMode,
  hydrateSubmissionMessages,
  getStartGenerationStreamId,
  buildOptimisticConversation,
  getGenerationProtocolVersion,
  GENERATION_PROTOCOL_VERSION,
  getStartGenerationCreatedAt,
  isUnnegotiatedV2StartControl,
  supportsGenerationProtocolV2,
  START_GENERATION_NETWORK_RETRIES,
  getStartGenerationConversationId,
  getPredecessorMismatchStartResponse,
  START_GENERATION_READINESS_TIMEOUT_MS,
  ResumableRun,
  documentVisibility,
  attachResumableStream,
} from '@librechat/chat';
import type {
  TMessage,
  TPayload,
  TSubmission,
  TConversation,
  TPendingSteer,
  EventSubmission,
  TSteerUpdatedEvent,
} from 'librechat-data-provider';
import type {
  ResumableHost,
  ResumableCache,
  SubscriptionTarget,
  StartGenerationResult,
  GenerationProtocolVersion,
} from '@librechat/chat';
import type { QueuedMessageOrigin, DrainAfterAbort, RunEnd } from '~/hooks/Chat/queue';
import type { ActiveJobsResponse, StreamStatusResponse } from '~/data-provider';
import type { EventHandlerParams } from './useEventHandlers';
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
    }),
    [addActiveJob, queryClient, removeActiveJob],
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
  const subscribeToStream = useCallback(
    (target: SubscriptionTarget) => {
      const host: ResumableHost = {
        logger,
        visibility: documentVisibility,
        fetchStatus: fetchStreamStatus,
        subscribe: (next) => subscribeToStream(next),
        cache: resumableCache,
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
      };
      attachResumableStream(run, host, target);
    },
    [
      run,
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

  /**
   * Start generation (POST request that returns streamId)
   * Posts through the host transport, which owns auth refresh.
   * Retries transient network failures and startup readiness responses.
   * Readiness retries honor Retry-After until cleanup or the readiness window expires.
   */
  const startGeneration = useCallback(
    async (
      currentSubmission: TSubmission,
      signal?: AbortSignal,
    ): Promise<StartGenerationResult | null> => {
      const payloadData = createPayload(currentSubmission);
      let { payload } = payloadData;
      payload = removeNullishValues(payload) as TPayload;

      clearStepMaps();

      const url = payloadData.server;

      let lastError: unknown = null;
      let requestAttempts = 0;
      let networkAttempts = 0;
      let readinessAttempts = 0;
      const readinessDeadline = run.scheduler.now() + START_GENERATION_READINESS_TIMEOUT_MS;

      while (!signal?.aborted) {
        const recoverySteerId = getRecoverySteerId(currentSubmission);
        const conversationId = currentSubmission.conversation?.conversationId;
        if (
          recoverySteerId != null &&
          conversationId &&
          jotaiStore.get(recoveryDispositionsFamily(conversationId))[recoverySteerId] != null
        ) {
          restoreQueuedSubmission(currentSubmission);
          errorHandler({
            data: getStreamStartFailureData(localize('com_ui_steer_recovery_held')),
            submission: currentSubmission as EventSubmission,
          });
          setShowStopButton(false);
          setIsSubmitting(false);
          setSubmission(null);
          return null;
        }
        requestAttempts += 1;
        try {
          const data = await transport.start({ server: url, payload }, { signal });
          if (signal?.aborted) {
            return null;
          }
          if (isSettledStartResponse(data)) {
            const conversationId = getStartGenerationConversationId(data);
            if (conversationId) {
              logger.log('ResumableSSE', 'Generation already settled:', { conversationId });
              return {
                status: 'settled',
                conversationId,
                generationProtocolVersion: GENERATION_PROTOCOL_VERSION,
              };
            }
            lastError = { response: { data } };
            break;
          }
          if (isReplacedStartResponse(data)) {
            const streamId = getStartGenerationStreamId(data);
            const conversationId = getStartGenerationConversationId(data);
            const generationCreatedAt = getStartGenerationCreatedAt(data);
            if (streamId && conversationId && generationCreatedAt != null) {
              return {
                status: 'replaced',
                streamId,
                conversationId,
                generationCreatedAt,
                generationProtocolVersion: GENERATION_PROTOCOL_VERSION,
              };
            }
            lastError = { response: { data } };
            break;
          }
          if (isUnnegotiatedV2StartControl(data)) {
            /** Never reinterpret a v2-only control word as a legacy stream
             * response merely because it also carries a stream id. */
            lastError = { response: { data } };
            break;
          }
          const streamId = getStartGenerationStreamId(data);
          if (streamId) {
            const resumed = isResumedStartResponse(data);
            const generationCreatedAt = getStartGenerationCreatedAt(data);
            const generationProtocolVersion = getGenerationProtocolVersion(data);
            logger.log('ResumableSSE', 'Generation started:', {
              streamId,
              resumed,
              generationCreatedAt,
              generationProtocolVersion,
            });
            return {
              status: 'stream',
              streamId,
              resumed,
              generationProtocolVersion,
              ...(generationCreatedAt != null && { generationCreatedAt }),
            };
          }

          lastError = { response: { data } };
          break;
        } catch (error) {
          if (signal?.aborted) {
            return null;
          }

          const predecessorMismatch = getPredecessorMismatchStartResponse(error);
          if (predecessorMismatch != null) {
            /** Admission was atomically rejected before this queued turn
             * created anything. Put the exact row back immediately and let
             * the caller reconcile or hand off to the generation that won. */
            restoreQueuedSubmission(currentSubmission);
            logger.log('ResumableSSE', 'Queued generation lost predecessor fence', {
              expectedPredecessorCreatedAt: currentSubmission.expectedPredecessorCreatedAt,
              ...predecessorMismatch,
            });
            return {
              status: 'predecessor_mismatch',
              ...predecessorMismatch,
              generationProtocolVersion: GENERATION_PROTOCOL_VERSION,
            };
          }

          lastError = error;
          const isNetworkError = isRetryableNetworkError(error);
          const isServerNotReady = isServerNotReadyError(error);
          const remainingReadinessMs = readinessDeadline - run.scheduler.now();
          const shouldRetryNetwork =
            isNetworkError && networkAttempts < START_GENERATION_NETWORK_RETRIES - 1;
          const shouldRetryServerNotReady = isServerNotReady && remainingReadinessMs > 0;

          if (shouldRetryNetwork || shouldRetryServerNotReady) {
            networkAttempts += isNetworkError ? 1 : 0;
            readinessAttempts += isServerNotReady ? 1 : 0;
            const fallbackDelay = Math.min(1000 * Math.pow(2, requestAttempts - 1), 8000);
            const retryDelay = isServerNotReady
              ? Math.min(getRetryAfterDelay(error, fallbackDelay), remainingReadinessMs)
              : fallbackDelay;
            const reason = isServerNotReady ? 'Server not ready' : 'Network error';
            const attempt = isServerNotReady ? readinessAttempts : networkAttempts;
            const limit = isServerNotReady
              ? `${Math.ceil(START_GENERATION_READINESS_TIMEOUT_MS / 1000)}s readiness window`
              : `${START_GENERATION_NETWORK_RETRIES}`;
            logger.log(
              'ResumableSSE',
              `${reason} starting generation, retrying in ${retryDelay}ms (attempt ${attempt}/${limit})`,
            );
            const shouldContinue = await waitForRetryDelay(retryDelay, signal);
            if (!shouldContinue) {
              return null;
            }
            continue;
          }

          // Don't retry: either not a network error or max retries reached
          break;
        }
      }

      if (signal?.aborted) {
        return null;
      }

      logger.error('ResumableSSE', 'Error starting generation:', lastError);

      const startError = toStartGenerationError(lastError);
      const errorData = startError?.response?.data;
      const responseStatus = startError?.response?.status;
      if (responseStatus != null && responseStatus >= 400 && responseStatus < 500) {
        const recoverySteerId = getRecoverySteerId(currentSubmission);
        const conversationId = currentSubmission.conversation?.conversationId;
        const recoveryRejected =
          errorData != null &&
          typeof errorData === 'object' &&
          'code' in errorData &&
          (errorData.code === 'RECOVERY_PAYLOAD_MISMATCH' ||
            errorData.code === 'INVALID_RECOVERY_REQUEST');
        if (recoveryRejected && recoverySteerId != null && conversationId) {
          jotaiStore.set(recoveryDispositionsFamily(conversationId), (previous) =>
            blockRecovery(previous, recoverySteerId),
          );
        }
        // The server rejected admission before exposing a generation. Restore
        // the exact queue row/position; ambiguous transport/5xx outcomes must
        // first reconcile durable state instead of risking a duplicate start.
        restoreQueuedSubmission(currentSubmission);
      } else {
        const recoverySteerId = getRecoverySteerId(currentSubmission);
        const recoveryConvoId = currentSubmission.conversation?.conversationId;
        if (recoverySteerId != null && recoveryConvoId) {
          try {
            const status = await fetchStreamStatus(recoveryConvoId);
            if (signal?.aborted) {
              return null;
            }
            const releasedSource = status.unrecoveredSteers?.find(
              (steer) => steer.steerId === recoverySteerId,
            );
            if (
              status.active === false &&
              supportsGenerationProtocolV2(status) &&
              releasedSource != null
            ) {
              convertSteersToQueued(recoveryConvoId, [releasedSource], {
                generationProtocolVersion: GENERATION_PROTOCOL_VERSION,
                allowPreviouslyConvertedIds: [recoverySteerId],
              });
            }
          } catch (error) {
            if (signal?.aborted) {
              return null;
            }
            logger.warn('ResumableSSE', 'Could not recover source after start failure', {
              conversationId: recoveryConvoId,
              error,
            });
          }
        }
      }
      if (signal?.aborted) {
        return null;
      }
      const termsRefused = reopenTermsOnRefusal(queryClient, errorData);
      errorHandler({
        data: getStreamStartFailureData(
          errorData,
          termsRefused ? localize('com_error_terms_not_accepted') : undefined,
        ),
        submission: currentSubmission as EventSubmission,
      });
      setShowStopButton(false);
      setIsSubmitting(false);
      setSubmission(null);
      return null;
    },
    [
      run,
      clearStepMaps,
      convertSteersToQueued,
      errorHandler,
      jotaiStore,
      localize,
      queryClient,
      restoreQueuedSubmission,
      setIsSubmitting,
      setShowStopButton,
      setSubmission,
      transport,
    ],
  );

  useEffect(() => {
    if (!submission || Object.keys(submission).length === 0) {
      logger.log('ResumableSSE', 'No submission, cleaning up');
      // Clear reconnect timeout if submission is cleared
      run.clearReconnectTimeout();
      run.stopForegroundReattach?.();
      run.stopForegroundReattach = null;
      // Close SSE but do NOT dispatch cancel - navigation should not abort
      run.stream?.abort();
      run.stream = null;
      setStreamId(null);
      run.reconnectAttempt = 0;
      run.submission = null;
      return;
    }

    const resumableSubmission = submission as TSubmission & {
      resumeStreamId?: string;
      resumeGenerationCreatedAt?: number;
      resumeGenerationProtocolVersion?: GenerationProtocolVersion;
    };
    const resumeStreamId = resumableSubmission.resumeStreamId;
    const resumeGenerationCreatedAt = resumableSubmission.resumeGenerationCreatedAt;
    const resumeGenerationProtocolVersion =
      resumableSubmission.resumeGenerationProtocolVersion ?? 1;
    logger.log('ResumableSSE', 'Effect triggered', {
      conversationId: submission.conversation?.conversationId,
      hasResumeStreamId: !!resumeStreamId,
      resumeStreamId,
      userMessageId: submission.userMessage?.messageId,
    });

    run.submission = submission;
    run.runEndPublished = false;
    run.identityConfirmed = false;
    run.subscribedEpoch = null;
    if (submission.conversation?.conversationId != null) {
      /** A different run starting here owns the conversation's end from now on: an earlier run
       *  this pane left is superseded, and a Stop belonged to that earlier run. Reattaching to
       *  the same generation keeps both, since that run's end is still owed and it was stopped. */
      const startingConvoId = submission.conversation.conversationId;
      const sameGeneration = (generationCreatedAt?: number) =>
        resumeStreamId != null &&
        resumeGenerationCreatedAt != null &&
        generationCreatedAt === resumeGenerationCreatedAt;
      const stopFamily = stopRequestedByConvoId(startingConvoId);
      const stopped = jotaiStore.get(stopFamily);
      if (stopped != null && !sameGeneration(stopped.generationCreatedAt)) {
        jotaiStore.set(stopFamily, null);
      }
      const detachedFamily = detachedRunByConvoId(startingConvoId);
      const detached = jotaiStore.get(detachedFamily);
      if (detached != null && !sameGeneration(detached.generationCreatedAt)) {
        jotaiStore.set(detachedFamily, null);
      }
    }
    const startController = new AbortController();
    const { signal } = startController;
    const isCurrentEffect = () => !signal.aborted && run.submission === submission;

    const initStream = async () => {
      if (!isCurrentEffect()) {
        return;
      }

      bindResponse(submission);
      setIsSubmitting(true);
      /** Starting a new generation is the one interval where `isSubmitting`
       *  is true but no generation epoch exists yet. Clear any prior epoch and
       *  withhold live Stop/steer controls; the composer can still queue input.
       *  subscribeToStream exposes controls once the POST/resume installs the
       *  server-provided epoch. */
      const submissionConversationId =
        submission.conversation?.conversationId ?? Constants.NEW_CONVO;
      if (!resumeStreamId || resumeGenerationCreatedAt == null) {
        updateActiveGenerationCreatedAt(resumeStreamId ?? submissionConversationId, null);
      }
      setShowStopButton(false);

      if (resumeStreamId) {
        if (!isCurrentEffect()) {
          return;
        }
        // Resume: just subscribe to existing stream, don't start new generation
        logger.log('ResumableSSE', 'Resuming existing stream:', resumeStreamId);
        setStreamId(resumeStreamId);
        // Optimistically add to active jobs (in case it's not already there)
        addActiveJob(resumeStreamId);
        subscribeToStream({
          streamId: resumeStreamId,
          submission,
          isResume: true,
          generationCreatedAt: resumeGenerationCreatedAt,
          generationProtocolVersion: resumeGenerationProtocolVersion,
          signal,
        });
      } else {
        // New generation: start and then subscribe
        logger.log('ResumableSSE', 'Starting NEW generation');
        const startResult = await startGeneration(submission, signal);
        if (!isCurrentEffect()) {
          return;
        }
        if (startResult) {
          let replacementStart = startResult.status === 'replaced' ? startResult : null;
          let knownReplacementStatus: StreamStatusResponse | null = null;

          if (startResult.status === 'predecessor_mismatch') {
            const expectedPredecessorCreatedAt = submission.expectedPredecessorCreatedAt;
            let refreshedStatus: StreamStatusResponse | null = null;
            try {
              refreshedStatus = await fetchStreamStatus(startResult.conversationId);
              if (!isCurrentEffect()) {
                return;
              }
            } catch (error) {
              if (!isCurrentEffect()) {
                return;
              }
              logger.warn('ResumableSSE', 'Could not refresh predecessor mismatch status', {
                conversationId: startResult.conversationId,
                error,
              });
            }
            if (!isCurrentEffect()) {
              return;
            }

            const reportedReplacement =
              startResult.active === true &&
              startResult.generationCreatedAt != null &&
              startResult.generationCreatedAt !== expectedPredecessorCreatedAt;

            if (
              refreshedStatus?.active === true &&
              supportsGenerationProtocolV2(refreshedStatus) &&
              typeof refreshedStatus.streamId === 'string' &&
              refreshedStatus.streamId.length > 0 &&
              refreshedStatus.createdAt != null &&
              refreshedStatus.createdAt !== expectedPredecessorCreatedAt
            ) {
              replacementStart = {
                status: 'replaced',
                streamId: refreshedStatus.streamId,
                conversationId: startResult.conversationId,
                generationCreatedAt: refreshedStatus.createdAt,
                generationProtocolVersion: GENERATION_PROTOCOL_VERSION,
              };
              knownReplacementStatus = refreshedStatus;
              restoreQueuedSubmission(submission, refreshedStatus.createdAt);
            } else if (reportedReplacement) {
              /** The atomic 409 is itself authoritative at admission time.
               * Status publication can lag that response, so retain its exact
               * epoch as a minimal handoff snapshot; the fenced SSE attach can
               * reconcile if the replacement finishes in the meantime. */
              replacementStart = {
                status: 'replaced',
                streamId: startResult.streamId,
                conversationId: startResult.conversationId,
                generationCreatedAt: startResult.generationCreatedAt!,
                generationProtocolVersion: GENERATION_PROTOCOL_VERSION,
              };
              knownReplacementStatus = {
                active: true,
                status: 'running',
                streamId: startResult.streamId,
                createdAt: startResult.generationCreatedAt,
                generationProtocolVersion: GENERATION_PROTOCOL_VERSION,
              };
              restoreQueuedSubmission(submission, startResult.generationCreatedAt!);
            } else {
              /** No different live epoch is proven. Reconcile away the
               * optimistic C rows, but publish no run-end: doing so would
               * immediately drain and re-send the just-restored queue item. */
              const terminalWinnerCreatedAt =
                refreshedStatus?.createdAt ?? startResult.generationCreatedAt;
              if (terminalWinnerCreatedAt != null) {
                restoreQueuedSubmission(submission, terminalWinnerCreatedAt);
              }
              if (refreshedStatus != null) {
                queryClient.setQueryData(
                  streamStatusQueryKey(startResult.conversationId),
                  refreshedStatus,
                );
              } else {
                queryClient.invalidateQueries({
                  queryKey: streamStatusQueryKey(startResult.conversationId),
                });
              }
              try {
                await queryClient.invalidateQueries({
                  queryKey: [QueryKeys.messages, startResult.conversationId],
                  refetchType: 'all',
                });
                if (!isCurrentEffect()) {
                  return;
                }
                await queryClient.invalidateQueries({
                  queryKey: [QueryKeys.conversation, startResult.conversationId],
                  refetchType: 'all',
                });
                if (!isCurrentEffect()) {
                  return;
                }
                await invalidateConversationLists(queryClient);
                if (!isCurrentEffect()) {
                  return;
                }
                await queryClient.invalidateQueries({ queryKey: [QueryKeys.pinnedConversations] });
                if (!isCurrentEffect()) {
                  return;
                }
              } catch (error) {
                if (!isCurrentEffect()) {
                  return;
                }
                logger.warn('ResumableSSE', 'Could not reconcile predecessor mismatch history', {
                  conversationId: startResult.conversationId,
                  error,
                });
              }
              clearDrainAfterAbort(startResult.conversationId, expectedPredecessorCreatedAt);
              run.submission = null;
              setIsSubmitting(false);
              setShowStopButton(false);
              setSubmission(null);
              return;
            }
          }

          if (replacementStart != null) {
            /** This POST belongs to an older optimistic submission, while the
             * conversation-scoped stream id now belongs to a newer generation.
             * Never attach A's submission directly to B. Discover B's
             * authoritative resume snapshot, refresh durable history, then let
             * useResumeOnLoad construct a new epoch-fenced submission. */
            const replacementStatus =
              knownReplacementStatus ??
              (await fetchStreamStatus(replacementStart.conversationId).catch((error) => {
                if (!isCurrentEffect()) {
                  return null;
                }
                logger.warn('ResumableSSE', 'Could not discover replacement start generation', {
                  conversationId: replacementStart.conversationId,
                  generationCreatedAt: replacementStart.generationCreatedAt,
                  error,
                });
                return null;
              }));
            if (!isCurrentEffect()) {
              return;
            }
            if (
              replacementStatus?.active === true &&
              supportsGenerationProtocolV2(replacementStatus) &&
              replacementStatus.streamId === replacementStart.streamId &&
              replacementStatus.createdAt === replacementStart.generationCreatedAt
            ) {
              const startedAsNewConvo = isInitialNewConversation(submission);
              try {
                await queryClient.invalidateQueries({
                  queryKey: [QueryKeys.messages, replacementStart.conversationId],
                  refetchType: 'all',
                });
                if (!isCurrentEffect()) {
                  return;
                }
                await queryClient.invalidateQueries({
                  queryKey: [QueryKeys.conversation, replacementStart.conversationId],
                  refetchType: 'all',
                });
                if (!isCurrentEffect()) {
                  return;
                }
                await invalidateConversationLists(queryClient);
                if (!isCurrentEffect()) {
                  return;
                }
                await queryClient.invalidateQueries({ queryKey: [QueryKeys.pinnedConversations] });
                if (!isCurrentEffect()) {
                  return;
                }
              } catch (error) {
                if (!isCurrentEffect()) {
                  return;
                }
                logger.warn('ResumableSSE', 'Could not refresh replacement generation history', {
                  conversationId: replacementStart.conversationId,
                  error,
                });
              }
              const authoritativeConversation = queryClient.getQueryData<TConversation>([
                QueryKeys.conversation,
                replacementStart.conversationId,
              ]);
              if (authoritativeConversation?.conversationId === replacementStart.conversationId) {
                const localConversationId = isInitialNewConversation(submission)
                  ? Constants.NEW_CONVO
                  : replacementStart.conversationId;
                setConversation?.((current) =>
                  keepLocalCodeApprovalMode(
                    authoritativeConversation,
                    current,
                    localConversationId,
                  ),
                );
              }
              queryClient.setQueryData(streamStatusQueryKey(replacementStart.conversationId), {
                ...replacementStatus,
                generationHandoff: true,
              });
              updateActiveGenerationCreatedAt(
                replacementStart.conversationId,
                replacementStart.generationCreatedAt,
                undefined,
                replacementStart.generationProtocolVersion,
              );
              addActiveJob(replacementStart.streamId);
              if (startedAsNewConvo) {
                run.optimisticStreamIds.add(replacementStart.streamId);
                updateActiveGenerationCreatedAt(
                  Constants.NEW_CONVO,
                  replacementStart.generationCreatedAt,
                  undefined,
                  replacementStart.generationProtocolVersion,
                );
                replaceNewConversationUrl(replacementStart.conversationId);
              }
              seedSteerChips(
                replacementStart.conversationId,
                (replacementStatus.resumeState?.pendingSteers ?? []) as TPendingSteer[],
                replacementStart.generationCreatedAt,
                replacementStart.generationProtocolVersion,
              );
              settleAppliedSteerParts(
                replacementStart.conversationId,
                replacementStatus.resumeState?.aggregatedContent ??
                  replacementStatus.aggregatedContent,
              );
              run.replacementHandoff = true;
              run.submission = null;
              setIsSubmitting(true);
              setShowStopButton(true);
              setSubmission(null);
              return;
            }

            /** The server reported a replacement but its authoritative status
             * could not be confirmed. Drop A without ever attaching it to B;
             * leave history/status invalidated so navigation/reload can recover,
             * and keep any queued text parked (non-completed outcome). */
            queryClient.invalidateQueries({
              queryKey: streamStatusQueryKey(replacementStart.conversationId),
            });
            queryClient.invalidateQueries({
              queryKey: [QueryKeys.messages, replacementStart.conversationId],
            });
            clearDrainAfterAbort(
              replacementStart.conversationId,
              replacementStart.generationCreatedAt,
            );
            setIsSubmitting(false);
            setShowStopButton(false);
            setRunEnd({
              conversationId: replacementStart.conversationId,
              outcome: 'error',
              endedAt: run.scheduler.now(),
              generationCreatedAt: replacementStart.generationCreatedAt,
            });
            setSubmission(null);
            return;
          }
          if (startResult.status === 'predecessor_mismatch') {
            // Every mismatch path above either hands off or reconciles.
            return;
          }
          if (startResult.status === 'settled') {
            const settledConversationId = startResult.conversationId;
            const startedAsNewConvo = isInitialNewConversation(submission);
            let persistedConversation: TConversation | undefined;
            let conversationLookup: 'found' | 'not_found' | 'inconclusive' = 'inconclusive';
            try {
              persistedConversation = await fetchConvoSnapshot(
                queryClient,
                settledConversationId,
                () => dataService.getConversationById(settledConversationId),
              );
              if (!isCurrentEffect()) {
                return;
              }
              conversationLookup = 'found';
            } catch (error) {
              if (!isCurrentEffect()) {
                return;
              }
              conversationLookup =
                toStartGenerationError(error)?.response?.status === 404
                  ? 'not_found'
                  : 'inconclusive';
              logger.warn('ResumableSSE', 'Could not load settled generation conversation', {
                conversationId: settledConversationId,
                conversationLookup,
                error,
              });
            }

            if (persistedConversation != null) {
              /** The persisted copy carries the mode the settled turn started
               *  with; a pick made while the start was pending is newer. The
               *  live conversation still holds the new-chat id here because no
               *  stream event ran for this turn. */
              const settledCopy = persistedConversation;
              const localConversationId = startedAsNewConvo
                ? Constants.NEW_CONVO
                : settledConversationId;
              const conversationRecord = keepLocalCodeApprovalMode(
                settledCopy,
                queryClient.getQueryData<TConversation>([
                  QueryKeys.conversation,
                  settledConversationId,
                ]),
                settledConversationId,
              );
              queryClient.setQueryData(
                [QueryKeys.conversation, settledConversationId],
                conversationRecord,
              );
              upsertConvoInAllQueries(queryClient, conversationRecord, true, 'snapshot');
              setConversation?.((current) =>
                keepLocalCodeApprovalMode(settledCopy, current, localConversationId),
              );
              if (startedAsNewConvo) {
                replaceNewConversationUrl(settledConversationId);
              }
            }

            try {
              await queryClient.invalidateQueries({
                queryKey: [QueryKeys.messages, settledConversationId],
                refetchType: 'all',
              });
              if (!isCurrentEffect()) {
                return;
              }
              await queryClient.invalidateQueries({
                queryKey: [QueryKeys.conversation, settledConversationId],
                refetchType: 'all',
              });
              if (!isCurrentEffect()) {
                return;
              }
              await invalidateConversationLists(queryClient);
              if (!isCurrentEffect()) {
                return;
              }
              await queryClient.invalidateQueries({ queryKey: [QueryKeys.pinnedConversations] });
              if (!isCurrentEffect()) {
                return;
              }
            } catch (error) {
              if (!isCurrentEffect()) {
                return;
              }
              logger.warn('ResumableSSE', 'Could not refresh settled generation history', {
                conversationId: settledConversationId,
                error,
              });
            }

            if (conversationLookup === 'not_found' && startedAsNewConvo) {
              /** A failed/early-aborted first turn can settle without ever
               * creating a conversation. Never mint A's optimistic rows here:
               * prune any partial cache and return to `/c/new` instead of
               * leaving a phantom chat/URL. */
              removeConvoFromAllQueries(queryClient, settledConversationId);
              queryClient.removeQueries({
                queryKey: [QueryKeys.conversation, settledConversationId],
              });
              queryClient.removeQueries({
                queryKey: [QueryKeys.messages, settledConversationId],
              });
              newConversation?.({
                template: { conversationId: String(Constants.NEW_CONVO) },
              });
            }
            if (conversationLookup === 'inconclusive') {
              /** A network/5xx lookup cannot prove the conversation is absent.
               * Preserve its optimistic/cache state for a later refetch and do
               * not navigate or mint a fresh `/c/new` conversation. */
              clearDrainAfterAbort(settledConversationId, submission.expectedPredecessorCreatedAt);
            }
            clearStepMaps();
            setIsSubmitting(false);
            setShowStopButton(false);
            setRunEnd({
              conversationId: settledConversationId,
              outcome: conversationLookup === 'inconclusive' ? 'error' : 'aborted',
              startedAsNewConvo,
              endedAt: run.scheduler.now(),
            });
            setSubmission(null);
            return;
          }
          if (startResult.status !== 'stream') {
            return;
          }
          const {
            streamId: newStreamId,
            resumed,
            generationCreatedAt,
            generationProtocolVersion,
          } = startResult;
          setStreamId(newStreamId);
          // Optimistically add to active jobs
          addActiveJob(newStreamId);
          // Queue title generation if this is a new conversation (first message).
          // Skip temporary conversations: the server never generates titles for
          // them, so polling would 404 indefinitely.
          const isNewConvo = isInitialNewConversation(submission);
          if (isNewConvo && !submission.isTemporary) {
            queueTitleGeneration(newStreamId);
          }
          if (isInitialNewConversation(submission)) {
            run.optimisticStreamIds.add(newStreamId);
            replaceNewConversationUrl(newStreamId);
          }
          const streamSubmission = addOptimisticConversation(newStreamId, submission);
          run.submission = streamSubmission;
          // A deduped retry (status: 'resumed') attaches to an already-running stream, so
          // subscribe with resume=true to replay its state instead of only live events.
          subscribeToStream({
            streamId: newStreamId,
            submission: streamSubmission,
            isResume: resumed,
            generationCreatedAt,
            generationProtocolVersion,
            signal,
          });
        } else {
          logger.error('ResumableSSE', 'Failed to get streamId from startGeneration');
        }
      }
    };

    /* Fire-and-forget, but not silent: this sets the submitting flags before
       it does any work, so a throw would leave the composer generating with no
       stream, no final event and no way back but a reload. */
    initStream().catch((error: unknown) => {
      logger.error('[useResumableSSE] Failed to start the stream', error);
      setIsSubmitting(false);
      setShowStopButton(false);
      setSubmission(null);
    });

    return () => {
      logger.log('ResumableSSE', 'Cleanup - closing SSE, resetting UI state');
      startController.abort();
      // Cleanup on unmount/navigation - close connection but DO NOT abort backend
      // Reset UI state so it doesn't leak to other conversations
      // If user returns to this conversation, useResumeOnLoad will restore the state
      run.clearReconnectTimeout();
      run.cancelRetryFrames();
      run.stopForegroundReattach?.();
      run.stopForegroundReattach = null;
      // Reset reconnect counter before closing (so abort handler doesn't think we're reconnecting)
      run.reconnectAttempt = 0;
      /** The pane is moving off a run that is still generating (another chat, a new chat, or
       *  an unmount): the server keeps going with no subscriber and deletes the job when done,
       *  so no terminal event will reach the queue drain. Remember the run; on return its end is
       *  read from history. Only a run whose identity the server confirmed (its epoch, and the
       *  user message `created` or a resume's sync named) is remembered, and not a regeneration,
       *  a run that already published its end, or one the user stopped. */
      const closing = run.submission;
      const closingConvoId = closing?.conversation?.conversationId;
      const closingUserMessageId = closing?.userMessage?.messageId;
      const generationCreatedAt = run.subscribedEpoch;
      if (
        run.stream != null &&
        run.identityConfirmed &&
        !run.runEndPublished &&
        closing?.isRegenerate !== true &&
        closingConvoId != null &&
        closingConvoId !== Constants.NEW_CONVO &&
        closingUserMessageId != null &&
        generationCreatedAt != null &&
        !isStoppedRun(jotaiStore.get(stopRequestedByConvoId(closingConvoId)), generationCreatedAt)
      ) {
        jotaiStore.set(detachedRunByConvoId(closingConvoId), {
          userMessageId: closingUserMessageId,
          responseMessageId: closing?.initialResponse?.messageId,
          generationCreatedAt,
        });
      }
      run.stream?.abort();
      run.stream = null;
      // Clear handler maps to prevent memory leaks and stale state
      clearStepMaps();
      // Reset UI state on ordinary cleanup. A generation handoff already
      // cached the replacement and must remain non-idle until resume-on-load
      // installs its epoch-fenced submission.
      if (!run.replacementHandoff) {
        setIsSubmitting(false);
        setShowStopButton(false);
      }
      run.replacementHandoff = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [submission]);

  return { streamId };
}
