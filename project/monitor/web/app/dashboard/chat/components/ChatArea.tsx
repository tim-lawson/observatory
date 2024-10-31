import store, { RootState } from '@/app/store/store';
import { Button } from '@/components/ui/button';
import { useDispatch, useSelector } from 'react-redux';
import { TokenSelector } from './tokenselector';
import { v4 as uuidv4 } from 'uuid';
import {
  useClearConversationMutation,
  useLazySendMessageQuery,
  useRegisterInterventionMutation,
} from '@/app/store/api/chatApi';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  setChatTokens,
  setIsLoadingChat,
  setIsStreamingTokens,
  setMaxNewTokens,
  setTemperature,
} from '@/app/store/slices/chatSlice';
import {
  setDescriptionKeywordFilter,
  setGlobalNeuronFilter,
  setNeuronsMetadataDict,
  setSelectedAttributionToken,
  setSelectedTokenRange,
  setTableHighlightedNeuronIds,
} from '@/app/store/slices/neuronsSlice';
import { Neuron, NeuronPolarity } from '@/app/types/neuronData';
import { CornerDownLeft, HelpCircle, RefreshCcw, Trash, X } from 'lucide-react';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { toast } from '@/hooks/use-toast';
import { Gift } from 'lucide-react';
import { BugPlay, Brain, Eye } from 'lucide-react';
import {
  useFetchNeuronsMetadataQuery,
  useLazyFetchNeuronsMetadataQuery,
} from '@/app/store/api/neuronsApi';
import LinterPanel from './LinterPanel';
import { Settings } from 'lucide-react'; // Add this import
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { Input } from '@/components/ui/input'; // Add this import
import {
  ActivationPercentileFilter,
  AttributionFilter,
  ComplexFilter,
} from '@/app/types/neuronFilters';
import { PRESET_FLOWS, PresetFlow } from '@/app/types/presetFlow';
import {
  setFlowState,
  setShowChatArea,
  setShowNeuronsPanel,
  setShowSteeringPanel,
} from '@/app/store/slices/uiStateSlice';
import SteeringPanel from './SteeringPanel';
import {
  addLinterMessage,
  CustomLinterMessage,
  setLoadingTokenSelectionLinterMessageId,
  setSelectedClusterId,
  TokenSelectionLinterMessage,
  updateLinterMessage,
} from '@/app/store/slices/aiLinterSlice';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import { usePostHog } from 'posthog-js/react';

export default function ChatArea({ showPanel }: { showPanel: boolean }) {
  /**
   * Global state
   */
  const dispatch = useDispatch();
  const sessionId = useSelector((state: RootState) => state.uiState.sessionId);
  const chatTokens = useSelector((state: RootState) => state.chat.tokens);
  const isLoadingChat = useSelector(
    (state: RootState) => state.chat.isLoadingChat
  );
  const steeringSpecs = useSelector(
    (state: RootState) => state.steering.steeringSpecs
  );
  const neuronsMetadataDict = useSelector(
    (state: RootState) => state.neurons.metadataDict
  );
  const neuronDisplayModulation = useSelector(
    (state: RootState) => state.neurons.displayModulation
  );
  const generationParameters = useSelector(
    (state: RootState) => state.chat.generationParameters
  );
  const neurons: Neuron[] = useMemo(() => {
    return Object.values(neuronsMetadataDict?.run ?? {}).map((cur) => ({
      layer: cur.layer,
      neuron: cur.neuron,
      token: cur.token,
      polarity: NeuronPolarity.POS,
    }));
  }, [neuronsMetadataDict]);
  const selectedTokenRange = useSelector(
    (state: RootState) => state.neurons.displayModulation.selectedTokenRange
  );
  const selectedAttributionToken = useSelector(
    (state: RootState) =>
      state.neurons.displayModulation.selectedAttributionToken
  );
  const globalNeuronFilter = useSelector(
    (state: RootState) => state.neurons.globalNeuronFilter
  );

  /**
   * Local state
   */
  const [inputValue, setInputValue] = useState('');

  /**
   * Functions for sending messages and updating chat tokens
   */
  const [
    sendMessage,
    { data: sendMessageData, isFetching: isStreamingSendMessage },
  ] = useLazySendMessageQuery();
  useEffect(() => {
    if (sendMessageData) {
      dispatch(setChatTokens(sendMessageData));
    }
  }, [sendMessageData]);
  const inputDisabled = useMemo(() => {
    return sessionId === undefined || isLoadingChat;
  }, [sessionId, isLoadingChat]);

  /**
   * Reset neuron display
   */
  const resetNeuronModulation = () => {
    dispatch(setSelectedTokenRange(undefined));
    dispatch(setSelectedAttributionToken(undefined));
    dispatch(setDescriptionKeywordFilter(undefined));
    dispatch(setSelectedClusterId(undefined));
    dispatch(setTableHighlightedNeuronIds(undefined));
  };

  /**
   * Sending and receiving messages
   */
  const [interventionId, setInterventionId] = useState<string | undefined>(
    undefined
  );
  const [
    registerIntervention,
    { data: remoteInterventionId, isLoading: isRegisteringIntervention },
  ] = useRegisterInterventionMutation();
  useEffect(() => {
    dispatch(setIsStreamingTokens(isStreamingSendMessage));
    dispatch(
      setIsLoadingChat(isStreamingSendMessage || isRegisteringIntervention)
    );
  }, [isStreamingSendMessage, isRegisteringIntervention]);
  // When steering specs change (and there is actually steering to do), register the intervention
  // If there's nothing to do, clear interventionId
  // Invariant: interventionId should always be defined if steeringSpecs is defined and non-empty
  // Invariant: interventionId should always be undefined if steeringSpecs is undefined or empty
  useEffect(() => {
    if (steeringSpecs === undefined || steeringSpecs.length === 0) {
      setInterventionId(undefined);
    } else if (sessionId) {
      const params = {
        session_id: sessionId,
        interventions: steeringSpecs
          .filter((spec) => spec.isSteering)
          .map((spec) => ({
            token_ranges: spec.tokenRanges,
            filter: spec.filter,
            strength: spec.strength,
          })),
      };
      registerIntervention(params);
    }
  }, [sessionId, steeringSpecs]);
  useEffect(() => {
    if (remoteInterventionId) {
      setInterventionId(remoteInterventionId);
    }
  }, [remoteInterventionId]);
  // Trigger message sending once the intervention is registered, or if it was cleared
  useEffect(() => {
    // Refreshes are only okay if we have a session ID and chat tokens already
    if (sessionId && chatTokens !== undefined && !isStreamingSendMessage) {
      sendMessage({
        sessionId,
        interventionId,
        maxNewTokens: generationParameters.maxNewTokens,
        temperature: generationParameters.temperature,
        uuid: uuidv4(),
      });
      resetNeuronModulation();
    }
  }, [sessionId, interventionId]);

  /**
   * Fetching neuron metadata
   * (Metadata is always fetched when the global neuron filter is changed)
   */
  const [fetchNeuronsMetadata, { data: neuronsMetadataDictData }] =
    useLazyFetchNeuronsMetadataQuery();
  useEffect(() => {
    if (sessionId && globalNeuronFilter !== undefined) {
      fetchNeuronsMetadata({
        sessionId,
        filter: globalNeuronFilter,
      });
    }
  }, [globalNeuronFilter]);
  useEffect(() => {
    if (neuronsMetadataDictData) {
      dispatch(setNeuronsMetadataDict(neuronsMetadataDictData));
    }
  }, [neuronsMetadataDictData]);

  /**
   * When streaming is done:
   * - Trigger a refresh of the global filter
   * - Show token highlight hint if we're in a preset flow
   */
  const flowState = useSelector((state: RootState) => state.uiState.flowState);
  const presetFlow = useMemo(() => {
    if (flowState && PRESET_FLOWS[flowState.presetFlowId]) {
      return PRESET_FLOWS[flowState.presetFlowId];
    }
    return undefined;
  }, [flowState]);
  const flowStateRef = useRef(flowState);
  useEffect(() => {
    flowStateRef.current = flowState;
  }, [flowState]);
  useEffect(() => {
    if (!isLoadingChat && chatTokens !== undefined) {
      // Trigger a refresh so neurons are re-fetched with new state
      // TODO this is unnecessary right now because the neuron activations are not recomputed!
      // So we're just going to prevent the re-fetch for now.
      if (
        globalNeuronFilter !== undefined &&
        neuronsMetadataDict === undefined // Fetch if we don't have anything yet; just no re-fetching
      ) {
        dispatch(setGlobalNeuronFilter({ ...globalNeuronFilter }));
      }

      // Always show token highlight if it hasn't been shown before
      if (
        flowStateRef.current &&
        flowStateRef.current.showTokenHighlight === undefined
      ) {
        dispatch(
          setFlowState({ ...flowStateRef.current, showTokenHighlight: true })
        );
      }
    }
  }, [isLoadingChat, chatTokens]);

  const posthog = usePostHog();

  /**
   * Handle chat submit
   */
  const handleChatSubmit = (message: string | null = inputValue) => {
    if (message !== null && message === '') {
      toast({
        title: 'No message',
        description: 'Please enter a message to send.',
      });
      return;
    }
    if (sessionId === undefined) {
      toast({
        title: 'Waiting to connect to server',
        description: 'Please wait while we connect to our server.',
      });
      return;
    }

    if (!isStreamingSendMessage) {
      sendMessage({
        sessionId,
        message: message ?? undefined,
        interventionId, // Recall the invariances; this should always be accurate (TODO resolve a race condition that I think exists)
        maxNewTokens: generationParameters.maxNewTokens,
        temperature: generationParameters.temperature,
        uuid: uuidv4(),
      });
      resetNeuronModulation();
      setInputValue('');

      // PostHog event tracking
      posthog.capture('chat_message_sent', {
        sessionId: sessionId,
        message: message,
        message_length: message?.length ?? 0,
        max_new_tokens: generationParameters.maxNewTokens,
        temperature: generationParameters.temperature,
      });
    }

    if (!presetFlow && !showNeuronsPanel) {
      dispatch(setShowNeuronsPanel(true));
    }
    if (!presetFlow && !showSteeringPanel) {
      dispatch(setShowSteeringPanel(true));
    }
  };

  /**
   * Handle clear conversation
   */
  const [clearConversation, { isLoading: isClearingConversation }] =
    useClearConversationMutation();
  const handleClearConversation = () => {
    if (sessionId) {
      clearConversation(sessionId);
      dispatch(setChatTokens(undefined));
      resetNeuronModulation();

      // PostHog event tracking
      posthog.capture('conversation_cleared', {
        sessionId: sessionId,
      });
    }
  };

  /**
   * When a preset flow is selected, send the message
   */
  const [prevPresetFlowId, setPrevPresetFlowId] = useState<string | undefined>(
    undefined
  );
  useEffect(() => {
    if (
      flowState?.presetFlowId &&
      flowState.presetFlowId !== prevPresetFlowId
    ) {
      const presetFlow = PRESET_FLOWS[flowState.presetFlowId];
      if (!presetFlow) {
        return;
      }

      handleChatSubmit(presetFlow.prompt);
      dispatch(setShowChatArea(true));
      setPrevPresetFlowId(flowState.presetFlowId);
    }
  }, [flowState]);

  /**
   * Handle token selection in the ChatArea
   */
  // If in a preset flow, we can highlight custom tokens by default to hint the user
  const presetTokenHighlight = useMemo(() => {
    if (presetFlow && flowState?.showTokenHighlight) {
      return presetFlow.tokenHighlight;
    }
    return undefined;
  }, [flowState, presetFlow]);
  // Reset token selection when the selected token range is cleared
  const resetTokenSelectionRef = useRef<(() => void) | null>(null);
  useEffect(() => {
    if (selectedTokenRange === undefined) {
      resetTokenSelectionRef.current?.();
    }
  }, [selectedTokenRange]);
  // Store the id of the linter message to replace if the token selection changes while clusters are loading
  const [
    curLoadingTokenSelectionLinterMessageId,
    setCurLoadingTokenSelectionLinterMessageId,
  ] = useState<string | undefined>(undefined);

  const loadingTokenSelectionLinterMessageId = useSelector(
    (state: RootState) => state.aiLinter.loadingTokenSelectionLinterMessageId
  );

  const handleTokenSelectionChange = (range: [number, number]) => {
    console.log('handleTokenSelectionChange', range);
    if (chatTokens === undefined) {
      return;
    }

    // PostHog event tracking
    // TODO capture the token string
    posthog.capture('token_selected', {
      sessionId: sessionId,
      range_start: range[0],
      range_end: range[1],
    });

    // A token has been selected; we need to know for demo preset flows
    if (flowState) {
      dispatch(
        setFlowState({
          ...flowState,
          showTokenHighlight: false,
          tokensSelected: true,
        })
      );
    }

    if (neuronDisplayModulation.showNeuronsFrom === 'activation') {
      dispatch(setSelectedTokenRange(range));
    } else {
      if (neuronDisplayModulation.selectedAttributionToken === undefined) {
        // Trigger a refresh of the neurons
        const attrFilter: AttributionFilter = {
          type: 'attribution',
          target_token_idx: range[1] - 1,
          top_k: 1000,
        };
        // const actFilter: ComplexFilter = {
        //   type: 'complex',
        //   op: 'or',
        //   filters: [
        //     {
        //       type: 'activation_percentile',
        //       percentile: '1e-4',
        //       direction: 'bottom',
        //     },
        //     {
        //       type: 'activation_percentile',
        //       percentile: '1e-4',
        //       direction: 'top',
        //     },
        //   ],
        // };
        const curFilter: ComplexFilter = {
          type: 'complex',
          op: 'and',
          filters: [attrFilter],
        };

        dispatch(setSelectedAttributionToken(range[1]));
        dispatch(setGlobalNeuronFilter(curFilter));

        const messageId = loadingTokenSelectionLinterMessageId ?? uuidv4();
        const newMessage = {
          id: messageId,
          type: 'tokenSelection',
          tokenIdx: range[1],
          tokenString: chatTokens[range[1]].token,
          mode: 'attribution',
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

        // We don't want to keep the selection, so reset
        resetTokenSelectionRef.current?.();
      } else {
        dispatch(setSelectedTokenRange(range));
      }
    }
  };

  /**
   * Steering display state
   */
  const showSteeringPanel = useSelector(
    (state: RootState) => state.uiState.showSteeringPanel
  );
  const showNeuronsPanel = useSelector(
    (state: RootState) => state.uiState.showNeuronsPanel
  );

  /**
   * Auto-sticking messages to top or bottom
   */
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const [hasOverflow, setHasOverflow] = useState(false);
  const checkOverflow = useCallback(() => {
    if (messagesContainerRef.current) {
      const { scrollHeight, clientHeight } = messagesContainerRef.current;
      setHasOverflow(scrollHeight > clientHeight);
    }
  }, []);
  useEffect(() => {
    const resizeObserver = new ResizeObserver(checkOverflow);
    if (messagesContainerRef.current) {
      resizeObserver.observe(messagesContainerRef.current);
    }
    return () => {
      resizeObserver.disconnect();
    };
  }, [checkOverflow]);
  useEffect(() => {
    checkOverflow();
  }, [chatTokens, checkOverflow]);

  return (
    showPanel && (
      <>
        <div className="flex items-center space-x-2 mb-2">
          <div className="text-lg font-bold">
            Model Chat (Llama-3.1 8B Instruct)
          </div>
          {/* <Tooltip>
            <TooltipTrigger>
              <HelpCircle
                size={14}
                className="text-muted-foreground cursor-help"
              />
            </TooltipTrigger>
            <TooltipContent className="text-sm max-w-[30rem]">
              Use this panel to chat with the model. Ask it questions, then use
              the other panels to understand its outputs.
            </TooltipContent>
          </Tooltip> */}
        </div>
        <div className="relative flex flex-col flex-1 overflow-y-auto p-4 bg-gray-50 rounded-lg space-y-4">
          <div
            ref={messagesContainerRef}
            className={cn(
              'flex flex-1 overflow-y-auto text-sm',
              hasOverflow ? 'flex-col-reverse' : 'flex-col'
            )}
          >
            {chatTokens !== undefined && (
              <TokenSelector
                chatTokens={chatTokens}
                neurons={neurons}
                neuronsMetadataDict={neuronsMetadataDict}
                steeringSpecs={steeringSpecs}
                mousedOverNeurons={neuronDisplayModulation.mousedOverNeurons}
                showNeuronsFrom={neuronDisplayModulation.showNeuronsFrom}
                selectedAttributionToken={
                  neuronDisplayModulation.selectedAttributionToken
                }
                onSelectionChange={(ranges: [number, number][]) =>
                  handleTokenSelectionChange(ranges[0])
                }
                resetSelectionRef={resetTokenSelectionRef}
                allowMultipleSelections={false}
                presetTokenHighlight={presetTokenHighlight}
                ignoreFirstToken={true}
                disableUserSelection={isLoadingChat}
              />
            )}
          </div>
          <div>
            <form
              className="relative overflow-hidden rounded-lg border bg-background focus-within:ring-1 focus-within:ring-ring"
              onSubmit={(e) => {
                e.preventDefault();
                handleChatSubmit();
              }}
            >
              <fieldset disabled={inputDisabled}>
                <Textarea
                  id="message"
                  placeholder={
                    chatTokens === undefined
                      ? 'Send a message...'
                      : 'Continue chatting with the model...'
                  }
                  className="min-h-12 resize-none border-0 p-3 shadow-none focus-visible:ring-0 text-sm"
                  value={inputValue}
                  onChange={(e) => {
                    setInputValue(e.target.value);
                  }}
                  onKeyDown={(e: React.KeyboardEvent<HTMLTextAreaElement>) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleChatSubmit();
                    }
                  }}
                />
                <div className="flex items-center justify-end p-3 pt-0 space-x-2">
                  <span className="text-xs text-gray-500 font-mono">
                    llama-3.1-8b-instruct
                  </span>
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button
                        type="button"
                        size="sm"
                        variant="ghost"
                        className="p-0"
                      >
                        <Settings className="text-gray-500 size-4" />
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-72" side="top" align="end">
                      <div className="grid gap-4">
                        <div className="space-y-1">
                          <h4 className="text-sm font-medium leading-none">
                            Chat Settings
                          </h4>
                          <p className="text-xs text-muted-foreground">
                            Adjust token generation parameters.
                          </p>
                        </div>
                        <div className="grid gap-2">
                          <div className="grid grid-cols-3 items-center">
                            <Label htmlFor="max_tokens" className="text-xs">
                              Max Tokens
                            </Label>
                            <Input
                              id="max_tokens"
                              type="number"
                              defaultValue={generationParameters.maxNewTokens}
                              className="col-span-2 h-8"
                              onChange={(e) =>
                                dispatch(
                                  setMaxNewTokens(parseInt(e.target.value))
                                )
                              }
                            />
                          </div>
                          <div className="grid grid-cols-3 items-center">
                            <Label htmlFor="max_tokens" className="text-xs">
                              Temperature
                            </Label>
                            <Input
                              id="temperature"
                              type="number"
                              defaultValue={generationParameters.temperature}
                              className="col-span-2 h-8"
                              onChange={(e) =>
                                dispatch(
                                  setTemperature(parseFloat(e.target.value))
                                )
                              }
                            />
                          </div>
                        </div>
                      </div>
                    </PopoverContent>
                  </Popover>
                  {(!flowState || flowState.isDone) && (
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="gap-1.5 rounded-md"
                      disabled={chatTokens === undefined}
                      onClick={handleClearConversation}
                    >
                      Clear chat
                      <Trash className="size-3.5" />
                    </Button>
                  )}
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="gap-1.5 rounded-md"
                    disabled={chatTokens === undefined}
                    onClick={() => {
                      handleChatSubmit(null);
                      // PostHog event tracking
                      posthog.capture('regenerate_response', {
                        sessionId: sessionId,
                      });
                    }}
                  >
                    Regenerate
                    <RefreshCcw className="size-3.5" />
                  </Button>
                  <Button
                    type="submit"
                    size="sm"
                    className="gap-1.5 rounded-md"
                    disabled={inputValue === ''}
                  >
                    Send
                    <CornerDownLeft className="size-3.5" />
                  </Button>
                </div>
              </fieldset>
            </form>
          </div>
          {showSteeringPanel && (
            <>
              <hr />
              <SteeringPanel />
            </>
          )}
          <div className="absolute left-4 top-0 space-y-2">
            {selectedTokenRange && (
              <div className="rounded-lg bg-white p-2 max-w-xs shadow-sm">
                <div className="text-sm">Filtering to neurons firing at:</div>
                <span className="inline-flex items-center ml-2 bg-gray-200 text-secondary-foreground px-1.5 py-0.5 rounded-md text-xs shadow-sm border border-secondary-foreground/20">
                  Tokens [{selectedTokenRange[0]}, {selectedTokenRange[1]}]
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-3 ml-1 p-0"
                    onClick={() => {
                      dispatch(setSelectedTokenRange(undefined));
                    }}
                  >
                    <X size={12} />
                  </Button>
                </span>
              </div>
            )}
            {selectedAttributionToken && (
              <div className="rounded-lg bg-white p-2 max-w-lg shadow-sm">
                <div className="text-sm">Attributing to token:</div>
                <div className="inline-flex items-center ml-2 bg-orange-200 text-secondary-foreground px-1.5 py-0.5 rounded-md font-mono text-xs shadow-sm border border-secondary-foreground/20">
                  <div>
                    {chatTokens && chatTokens[selectedAttributionToken]
                      ? `"${chatTokens[selectedAttributionToken].token}"`
                      : selectedAttributionToken}
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-3 ml-1 p-0"
                    onClick={() => {
                      dispatch(setSelectedTokenRange(undefined));
                      dispatch(setSelectedAttributionToken(undefined));
                    }}
                  >
                    <X size={12} />
                  </Button>
                </div>
              </div>
            )}
            {neuronDisplayModulation.mousedOverTokenIndex !== undefined &&
              chatTokens &&
              chatTokens[neuronDisplayModulation.mousedOverTokenIndex]
                ?.top_log_probs && (
                <div className="rounded-lg bg-white p-2 max-w-xs shadow-sm">
                  <h4 className="text-sm font-semibold mb-2 text-gray-700">
                    Top probabilities:
                  </h4>
                  {chatTokens[
                    neuronDisplayModulation.mousedOverTokenIndex
                  ]?.top_log_probs?.map((el, index) => (
                    <div
                      key={index}
                      className="flex justify-between items-center text-sm mb-1"
                    >
                      <span className="font-mono text-xs bg-gray-100 px-1 py-0.5 rounded">
                        {el[0]}
                      </span>
                      <span className="text-gray-600">
                        {Math.exp(el[1]).toFixed(3)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
          </div>
        </div>
      </>
    )
  );
}
