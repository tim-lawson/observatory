import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import { BASE_URL } from '@/app/constants';
import { ChatToken } from '@/app/types/tokens';
import { EventSourcePolyfill } from 'event-source-polyfill';
import { useEffect } from 'react';
import { NeuronDBFilter, NeuronFilter } from '@/app/types/neuronFilters';
import { useState, useCallback } from 'react';

/**
 * Sending chat messages
 */

interface SendMessageParams {
  sessionId: string;
  message?: string;
  interventionId?: string;
  maxNewTokens?: number;
  temperature?: number;
  uuid: string; // Hack for cache invalidation
}

interface ChatTokenResponse {
  status: string;
  tokens?: ChatToken[];
}

export const sendMessageApi = createApi({
  reducerPath: 'sendMessageApi',
  baseQuery: () => ({ data: null }),
  endpoints: (builder) => ({
    sendMessageStreamingOutput: builder.query<ChatTokenResponse, SendMessageParams>({
      query: (params) => {
        const { sessionId, message, interventionId, maxNewTokens, temperature } = params;
        const url = new URL(`${BASE_URL}/message/${sessionId}`);

        if (message !== undefined) {
          url.searchParams.set('message', message);
        }
        if (interventionId !== undefined) {
          url.searchParams.set('intervention_id', interventionId);
        }
        if (maxNewTokens !== undefined) {
          url.searchParams.set('max_new_tokens', maxNewTokens.toString());
        }
        if (temperature !== undefined) {
          url.searchParams.set('temperature', temperature.toString());
        }
        return { url: url.toString() };
      },
      async onCacheEntryAdded(
        arg,
        { updateCachedData, cacheDataLoaded, cacheEntryRemoved, getCacheEntry }
      ) {
        const { sessionId, message, interventionId, maxNewTokens, temperature } = arg;
        const url = new URL(`${BASE_URL}/message/${sessionId}`);

        if (message !== undefined) {
          url.searchParams.set('message', message);
        }
        if (interventionId !== undefined) {
          url.searchParams.set('intervention_id', interventionId);
        }
        if (maxNewTokens !== undefined) {
          url.searchParams.set('max_new_tokens', maxNewTokens.toString());
        }
        if (temperature !== undefined) {
          url.searchParams.set('temperature', temperature.toString());
        }
        console.log('sending message', url.toString());

        const eventSource = new EventSourcePolyfill(url.toString());
        try {
          await cacheDataLoaded;

          eventSource.onmessage = (event) => {
            if (event.data === '[DONE]') {
              eventSource.close();
              updateCachedData((draft) => ({
                status: 'done',
                tokens: draft.tokens,
              }));
            } else {
              const newData = JSON.parse(event.data);
              updateCachedData((draft) => ({
                status: 'not_done',
                tokens: newData,
              }));
            }
          };

          eventSource.onerror = (error) => {
            eventSource.close();
          };

          // Wait for the cache entry to be removed (e.g., when the component unmounts)
          await cacheEntryRemoved;
          eventSource.close();
        } catch {
          eventSource.close();
        }
      },
    }),
  }),
});

const { useSendMessageStreamingOutputQuery, useLazySendMessageStreamingOutputQuery } = sendMessageApi;

export const useSendMessageQuery = (params: SendMessageParams) => {
  const { data, ...rest } = useSendMessageStreamingOutputQuery(params);
  const status = data?.status ?? 'not_done';
  return { ...rest, data: data?.tokens, isFetching: status === 'not_done' };
};

export const useLazySendMessageQuery = () => {
  const [trigger, result] = useLazySendMessageStreamingOutputQuery();
  const [hasTriggered, setHasTriggered] = useState(false);

  const wrappedTrigger = useCallback((...args: Parameters<typeof trigger>) => {
    setHasTriggered(true);
    return trigger(...args);
  }, [trigger]);

  const status = hasTriggered ? (result.data?.status ?? 'not_done') : 'idle';

  const lazyQueryResult = {
    ...result,
    data: result.data?.tokens,
    isFetching: status === 'not_done'
  };

  return [
    wrappedTrigger,
    lazyQueryResult
  ] as const;
};

/**
 * Interventions
 */

interface Intervention {
  token_ranges: [number, number][];
  filter: NeuronFilter;
  strength: number;
}

interface RegisterInterventionParams {
  session_id: string;
  interventions: Intervention[];
}

export const chatApi = createApi({
  reducerPath: 'chatApi',
  baseQuery: fetchBaseQuery({ baseUrl: BASE_URL }),
  endpoints: (builder) => ({
    registerIntervention: builder.mutation<string, RegisterInterventionParams>({
      query: ({ session_id, interventions }) => ({
        url: `register/intervention/${session_id}`,
        method: 'POST',
        body: interventions,
      }),
    }),
    clearConversation: builder.mutation<void, string>({
      query: (sessionId) => ({
        url: `message/clear/${sessionId}`,
        method: 'DELETE',
      }),
    }),
  }),
});

export const { useRegisterInterventionMutation, useClearConversationMutation } = chatApi;
