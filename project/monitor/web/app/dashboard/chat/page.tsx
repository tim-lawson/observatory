'use client';

import { v4 as uuidv4 } from 'uuid';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../../store/store';
import {
  setSessionId,
  setShowChatArea,
  setShowLinterPanel,
  setShowNeuronsPanel,
  setShowSteeringPanel,
} from '../../store/slices/uiStateSlice';
import { useFetchSessionQuery } from '@/app/store/api/sessionApi';
import { ResizableHandle, ResizablePanel } from '@/components/ui/resizable';
import { ResizablePanelGroup } from '@/components/ui/resizable';
import ChatArea from './components/ChatArea';
import NeuronDisplay from './components/NeuronDisplay';
import SteeringPanel from './components/SteeringPanel';
import {
  useLazySendMessageQuery,
  useRegisterInterventionMutation,
} from '@/app/store/api/chatApi';
import { setChatTokens, setIsLoadingChat } from '@/app/store/slices/chatSlice';
import { toast } from '@/hooks/use-toast';
import { PRESET_FLOWS } from '@/app/types/presetFlow';
import {
  DEFAULT_ACTIVATION_FILTER,
  setDescriptionKeywordFilter,
  setGlobalNeuronFilter,
  setSelectedAttributionToken,
  setSelectedTokenRange,
  setShowNeuronsFrom,
  setTableHighlightedNeuronIds,
} from '@/app/store/slices/neuronsSlice';
import { cn } from '@/lib/utils';
import LinterPanel from './components/LinterPanel';
import { ChatToken } from '@/app/types/tokens';
import {
  addLinterMessage,
  setLoadingTokenSelectionLinterMessageId,
  setSelectedClusterId,
  SwitchModeLinterMessage,
  TokenSelectionLinterMessage,
  updateLinterMessage,
} from '@/app/store/slices/aiLinterSlice';
import { motion } from 'framer-motion';
import { ComplexFilter } from '@/app/types/neuronFilters';

export default function ChatPage() {
  const dispatch = useDispatch();

  /**
   * Initialize session with ID
   */
  const [retryCount, setRetryCount] = useState(0);
  const sessionId = useSelector((state: RootState) => state.uiState.sessionId);
  const { data: newSessionId, error } = useFetchSessionQuery(undefined, {
    skip: sessionId !== undefined || retryCount >= 10,
    pollingInterval: 1000,
  });

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (!sessionId && !newSessionId) {
        setRetryCount((prev) => prev + 1);
        if (retryCount >= 10) {
          toast({
            title: 'Connection Failed',
            description:
              'Unable to connect to server after multiple attempts. Please refresh the page.',
            variant: 'destructive',
          });
        } else {
          toast({
            title: 'Warning',
            description:
              'Unable to connect to server. Please try again shortly.',
            variant: 'destructive',
          });
        }
      }
    }, 5000);

    if (newSessionId) {
      console.log('Initialized with sessionId:', newSessionId);
      toast({
        title: 'Connected to server',
        description: 'Successfully initialized new session',
      });
      dispatch(setSessionId(newSessionId));
      clearTimeout(timeoutId);
    }

    return () => clearTimeout(timeoutId);
  }, [newSessionId, dispatch, sessionId, retryCount]);

  /**
   * Panel display states
   */
  const showChatArea = useSelector(
    (state: RootState) => state.uiState.showChatArea
  );
  const showNeuronsPanel = useSelector(
    (state: RootState) => state.uiState.showNeuronsPanel
  );

  /**
   * If choose a new preset flow, go to the specified activation/attributionmode
   */
  const flowState = useSelector((state: RootState) => state.uiState.flowState);
  const [prevPresetFlowId, setPrevPresetFlowId] = useState<string | undefined>(
    undefined
  );
  useEffect(() => {
    if (
      flowState &&
      prevPresetFlowId !== flowState.presetFlowId &&
      PRESET_FLOWS[flowState.presetFlowId]
    ) {
      dispatch(
        setShowNeuronsFrom(
          PRESET_FLOWS[flowState.presetFlowId].initialShowNeuronsFrom
        )
      );
      setPrevPresetFlowId(flowState.presetFlowId);
    }
  }, [flowState, dispatch]);

  /**
   * Whenever showNeuronsFrom changes, reset the other state variables
   */
  const showNeuronsFrom = useSelector(
    (state: RootState) => state.neurons.displayModulation.showNeuronsFrom
  );
  const prevShowNeuronsFromRef = useRef(showNeuronsFrom);

  const loadingTokenSelectionLinterMessageId = useSelector(
    (state: RootState) => state.aiLinter.loadingTokenSelectionLinterMessageId
  );

  useEffect(() => {
    // Check if the value has actually changed
    if (prevShowNeuronsFromRef.current !== showNeuronsFrom) {
      dispatch(setSelectedTokenRange(undefined));
      dispatch(setSelectedAttributionToken(undefined));
      dispatch(setDescriptionKeywordFilter(undefined));
      dispatch(setSelectedClusterId(undefined));
      dispatch(setTableHighlightedNeuronIds(undefined));

      // If activation mode, reset the global neuron filter
      if (showNeuronsFrom === 'activation') {
        dispatch(setGlobalNeuronFilter(DEFAULT_ACTIVATION_FILTER));

        const messageId = loadingTokenSelectionLinterMessageId ?? uuidv4();
        const newMessage = {
          id: messageId,
          type: 'tokenSelection',
          mode: 'activation',
        } as TokenSelectionLinterMessage;

        if (loadingTokenSelectionLinterMessageId === undefined) {
          dispatch(addLinterMessage(newMessage));
          dispatch(setLoadingTokenSelectionLinterMessageId(messageId));
        } else {
          dispatch(
            updateLinterMessage({
              id: loadingTokenSelectionLinterMessageId,
              message: newMessage,
            })
          );
        }
      } else {
        dispatch(setGlobalNeuronFilter(undefined));
      }

      // Update the ref with the new value
      prevShowNeuronsFromRef.current = showNeuronsFrom;
    }
  }, [showNeuronsFrom, dispatch]);

  return (
    <>
      <div className="h-full flex">
        <motion.div
          initial={{ width: '0%' }}
          animate={{ width: showChatArea ? '40%' : '0%' }}
          exit={{ width: 0 }}
          transition={{ duration: 0.5 }}
          className={cn('flex flex-col h-full', showChatArea && 'p-4')}
        >
          <ChatArea showPanel={showChatArea} />
        </motion.div>
        {showChatArea && <div className="border-l" />}
        <motion.div
          className="flex flex-col justify-center items-center h-full"
          initial={{ width: '100%' }}
          animate={{ width: showChatArea ? '60%' : '100%' }}
          transition={{ duration: 0.5 }}
        >
          <motion.div
            initial={{ height: '100%', width: '60%' }}
            animate={{
              height: showNeuronsPanel ? '40%' : '100%',
              width: showChatArea ? '100%' : '60%',
            }}
            transition={{ duration: 0.5 }}
            className="p-4 flex flex-col"
          >
            <LinterPanel />
          </motion.div>
          {showNeuronsPanel && (
            <motion.div
              initial={{ height: '0%', width: '100%' }}
              animate={{
                height: showNeuronsPanel ? '60%' : '0%',
              }}
              transition={{ duration: 0.5 }}
              className="p-4 flex flex-col border-t"
            >
              <NeuronDisplay />
            </motion.div>
          )}
        </motion.div>
      </div>

      {/* Add the overlay for small screens */}
      <div className="hidden max-xl:flex fixed inset-0 bg-background items-center justify-center p-8 z-50">
        <div className="p-6 rounded-lg max-w-xl text-center">
          <h2 className="text-xl font-semibold mb-2">Screen Too Small</h2>
          <p className="text-muted-foreground">
            This interface requires a larger screen width.
            <br />
            Please try zooming out or using a larger screen.
          </p>
        </div>
      </div>
    </>
  );
}
