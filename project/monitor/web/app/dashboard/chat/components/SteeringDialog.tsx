import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { useEffect, useMemo, useState, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { v4 as uuidv4 } from 'uuid';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  FilterIcon,
  PauseIcon,
  PencilIcon,
  PlayIcon,
  SettingsIcon,
  TrashIcon,
  HelpCircle,
  EditIcon,
} from 'lucide-react';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '@/app/store/store';
import {
  addSteeringSpec,
  removeSteeringSpec,
  setSteeringSpec,
} from '@/app/store/slices/steeringSlice';
import { toast } from '@/hooks/use-toast';
import {
  useLazySendMessageQuery,
  useRegisterInterventionMutation,
} from '@/app/store/api/chatApi';
import { setChatTokens, setIsLoadingChat } from '@/app/store/slices/chatSlice';
import {
  Neuron,
  NeuronPolarity,
  NeuronsMetadataDict,
} from '@/app/types/neuronData';
import { useDebounce } from '@/hooks/use-debounce';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from '@/components/ui/resizable';
import { Label } from '@/components/ui/label';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { DataTable } from './datatable';
import { Slider } from '@/components/ui/slider';
import { TokenSelector } from './tokenselector';
import { columnsNoActivationSingleExplanation } from './columns';
import {
  useLazyFetchNeuronsAndMetadataQuery,
  useLazyFetchNeuronsMetadataQuery,
} from '@/app/store/api/neuronsApi';
import { NeuronDBFilter, SteeringSpec } from '@/app/types/neuronFilters';
import { PRESET_FLOWS } from '@/app/types/presetFlow';
import {
  setFlowState,
  setSteeringDialogSpec,
} from '@/app/store/slices/uiStateSlice';
import { cn } from '@/lib/utils';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { usePostHog } from 'posthog-js/react';

const DEFAULT_STRENGTHENING_TOP_K = 50;
const DEFAULT_DEACTIVATING_TOP_K = 500;
const MAX_TOP_K = 1000;
const DEFAULT_STRENGTH = 0.75;
const MAX_STRENGTH = 1.0;

export default function SteeringSpecDialog() {
  const dispatch = useDispatch();
  const steeringDialogSpec = useSelector(
    (state: RootState) => state.uiState.steeringDialogSpec
  );

  /**
   * Global state
   */
  const sessionId = useSelector((state: RootState) => state.uiState.sessionId);
  const steeringSpecs = useSelector(
    (state: RootState) => state.steering.steeringSpecs
  );
  const chatTokens = useSelector((state: RootState) => state.chat.tokens);
  const isLoadingChat = useSelector(
    (state: RootState) => state.chat.isLoadingChat
  );
  /**
   * NeuronDBFilter options for the user
   */
  const [concept, setConcept] = useState<string | null>(null);
  const [keyword, setKeyword] = useState<string | null>(null);
  const [polarity, setPolarity] = useState<NeuronPolarity | null>(null);
  const [topK, setTopK] = useState<number | null>(DEFAULT_STRENGTHENING_TOP_K);
  const [layerRange, setLayerRange] = useState<[number | null, number | null]>([
    null,
    null,
  ]);
  const [neuronRange, setNeuronRange] = useState<
    [number | null, number | null]
  >([null, null]);
  const [explanationScoreRange, setExplanationScoreRange] = useState<
    [number | null, number | null]
  >([null, null]);

  /**
   * Steering-related options
   */
  const [selectedTokenRanges, setSelectedTokenRanges] = useState<
    [number, number][] | null
  >(null);
  const [strength, setStrength] = useState<number>(DEFAULT_STRENGTH);

  /**
   * Update default options when a default steering spec is loaded in
   */
  const [prevSteeringDialogSpecId, setPrevSteeringDialogSpecId] = useState<
    string | undefined
  >(undefined);
  useEffect(() => {
    if (steeringDialogSpec && prevSteeringDialogSpecId === undefined) {
      const spec = steeringDialogSpec;
      if (spec && spec.filter.type === 'db') {
        setConcept(spec.filter.concept_or_embedding ?? null);
        setKeyword(spec.filter.keyword ?? null);
        setPolarity(spec.filter.polarity ?? null);
        setTopK(spec.filter.top_k ?? DEFAULT_STRENGTHENING_TOP_K);
        setLayerRange(spec.filter.layer_range ?? [null, null]);
        setNeuronRange(spec.filter.neuron_range ?? [null, null]);
        setExplanationScoreRange(
          spec.filter.explanation_score_range ?? [null, null]
        );
        setSelectedTokenRanges(spec.tokenRanges ?? null);
        setStrength(spec.strength ?? DEFAULT_STRENGTH);
      }
      setPrevSteeringDialogSpecId(spec.id);
    }
  }, [steeringDialogSpec]);

  /**
   * When the steering dialog is closed, reset the prev steering dialog spec id
   */
  useEffect(() => {
    if (steeringDialogSpec === null) {
      setPrevSteeringDialogSpecId(undefined);
    }
  }, [steeringDialogSpec]);

  /**
   * Code for fetching neurons
   */
  const [fetchNeurons, { data, isFetching: isFetchingNeurons }] =
    useLazyFetchNeuronsAndMetadataQuery();
  const neurons = useMemo(() => {
    return data?.neurons;
  }, [data]);
  const neuronsMetadataDict = useMemo(() => {
    return data?.metadata;
  }, [data]);

  const posthog = usePostHog();

  const handleApplyFilter = async () => {
    if (sessionId === undefined) {
      toast({
        title: 'No session ID',
        description: 'Could not find a session ID.',
      });
      return;
    }

    const dbFilter: NeuronDBFilter = {
      type: 'db',
      concept_or_embedding: debouncedConcept,
      keyword: debouncedKeyword,
      polarity: debouncedPolarity,
      top_k: debouncedTopK ?? DEFAULT_STRENGTHENING_TOP_K,
      layer_range: debouncedLayerRange,
      neuron_range: debouncedNeuronRange,
      explanation_score_range: debouncedExplanationScoreRange,
    };
    console.log('dbFilter', dbFilter);
    fetchNeurons({
      sessionId,
      filter: dbFilter,
    });
  };

  /**
   * Debounce the neuronDBFilter options and hit the DB when changes happen
   */
  const debounceTime = 250;
  const debouncedConcept = useDebounce(concept, debounceTime);
  const debouncedKeyword = useDebounce(keyword, debounceTime);
  const debouncedPolarity = useDebounce(polarity, debounceTime);
  const debouncedTopK = useDebounce(topK, debounceTime);
  const debouncedLayerRange = useDebounce(layerRange, debounceTime);
  const debouncedNeuronRange = useDebounce(neuronRange, debounceTime);
  const debouncedExplanationScoreRange = useDebounce(
    explanationScoreRange,
    debounceTime
  );
  useEffect(() => {
    if (sessionId) {
      handleApplyFilter();
    }
  }, [
    debouncedConcept,
    debouncedKeyword,
    debouncedPolarity,
    debouncedTopK,
    debouncedLayerRange,
    debouncedNeuronRange,
    debouncedExplanationScoreRange,
    sessionId,
  ]);

  /**
   * Handle UI events
   */
  const handleSubmit = () => {
    if (!selectedTokenRanges) {
      toast({
        title: 'No tokens selected',
        description: 'Please select at least one token.',
      });
      return;
    }

    posthog.capture('Submitted steering spec', {
      sessionId,
      concept,
      keyword,
      polarity,
      topK,
      layerRange,
      neuronRange,
      explanationScoreRange,
      selectedTokenRanges,
      strength,
    });

    const newFilter: NeuronDBFilter = {
      type: 'db',
      concept_or_embedding: concept,
      keyword,
      polarity,
      top_k: topK ?? DEFAULT_STRENGTHENING_TOP_K,
      layer_range: layerRange,
      neuron_range: neuronRange,
      explanation_score_range: explanationScoreRange,
    };

    // If the steering spec already exists, update it
    if (steeringSpecs && steeringDialogSpec) {
      const curSteeringSpec = steeringSpecs.find(
        (spec) => spec.id === steeringDialogSpec.id
      );
      if (curSteeringSpec) {
        dispatch(
          setSteeringSpec({
            id: curSteeringSpec.id,
            spec: {
              ...curSteeringSpec,
              filter: newFilter,
              tokenRanges: selectedTokenRanges,
              strength: strength,
            },
          })
        );
        dispatch(setSteeringDialogSpec(null));
        return;
      }
    }

    // Otherwise, create a new steering spec
    dispatch(
      addSteeringSpec({
        id: steeringDialogSpec?.id ?? uuidv4(),
        name: newFilter.concept_or_embedding ?? 'null',
        filter: newFilter,
        tokenRanges: selectedTokenRanges,
        isSteering: true,
        strength: strength,
      })
    );
    dispatch(setSteeringDialogSpec(null));
  };

  /**
   * Scroll the token selector to the bottom when the steering dialog is opened
   */
  const tokenSelectorRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const scrollToBottom = () => {
      if (tokenSelectorRef.current) {
        const lastChild = tokenSelectorRef.current.lastElementChild;
        if (lastChild) {
          lastChild.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
      }
    };
    const timeoutId = setTimeout(scrollToBottom, 100);
    return () => clearTimeout(timeoutId);
  }, [steeringDialogSpec]);

  return (
    <Dialog
      open={steeringDialogSpec !== null}
      onOpenChange={(open: boolean) => {
        if (!open) {
          dispatch(setSteeringDialogSpec(null));
          posthog.capture('Closed steering dialog', { sessionId });
        }
      }}
    >
      <DialogContent className="max-w-[90rem] p-6 flex h-[90vh] max-h-[90vh] flex-col">
        <DialogHeader>
          <DialogTitle>Select neurons to steer with</DialogTitle>
          {/* <DialogDescription>
            Adjust the neuron filters to select groups of neurons to steer; see
            the matching neurons update in real-time.
          </DialogDescription> */}
        </DialogHeader>
        <ResizablePanelGroup
          direction="horizontal"
          className="flex-grow h-full"
        >
          <ResizablePanel minSize={40} className="space-y-4 flex flex-col">
            <div>
              <div className="text-md font-medium">Neuron Filters</div>
              <div className="text-sm text-muted-foreground">
                Query our database of 450K neurons to find ones with specific
                behaviors.
              </div>
            </div>
            <div className="grid gap-4 mx-1">
              <div className="grid grid-cols-4 items-center gap-4">
                <Label
                  htmlFor="concept"
                  className="text-right flex items-center justify-end"
                >
                  Concept
                  <Tooltip>
                    <TooltipTrigger tabIndex={-1}>
                      <HelpCircle
                        size={14}
                        className="ml-2 text-muted-foreground cursor-help"
                      />
                    </TooltipTrigger>
                    <TooltipContent className="text-sm text-center max-w-[20rem]">
                      Enter a concept to find neurons related to it. This uses
                      semantic search to find relevant neurons.
                    </TooltipContent>
                  </Tooltip>
                </Label>
                <Input
                  id="concept"
                  placeholder="Enter concept for semantic search"
                  className="col-span-3"
                  value={concept ?? ''}
                  onChange={(e) => setConcept(e.target.value || null)}
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label
                  htmlFor="keyword"
                  className="text-right flex items-center justify-end"
                >
                  Keyword
                  <Tooltip>
                    <TooltipTrigger tabIndex={-1}>
                      <HelpCircle
                        size={14}
                        className="ml-2 text-muted-foreground cursor-help"
                      />
                    </TooltipTrigger>
                    <TooltipContent className="text-sm text-center max-w-[20rem]">
                      Enter a keyword for exact match (case-insensitive) on
                      neuron descriptions.
                    </TooltipContent>
                  </Tooltip>
                </Label>
                <Input
                  id="keyword"
                  placeholder="Enter keyword for exact match (case-insensitive)"
                  className="col-span-3"
                  value={keyword ?? ''}
                  onChange={(e) => setKeyword(e.target.value || null)}
                />
              </div>
              {/* <div className="grid grid-cols-4 items-center gap-4">
                <Label
                  htmlFor="polarity"
                  className="text-right flex items-center justify-end"
                >
                  Polarity

                    <Tooltip>
                      <TooltipTrigger>
                        <HelpCircle
                          size={14}
                          className="ml-2 text-muted-foreground cursor-help"
                        />
                      </TooltipTrigger>
                      <TooltipContent className="text-sm text-center max-w-[20rem]">
                        Choose the polarity of neurons: Positive (activating),
                        Negative (inhibiting), or All.
                      </TooltipContent>
                    </Tooltip>

                </Label>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" className="col-span-3">
                      {polarity === NeuronPolarity.POS
                        ? 'Positive'
                        : polarity === NeuronPolarity.NEG
                          ? 'Negative'
                          : 'All'}
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent>
                    <DropdownMenuItem
                      onSelect={() => setPolarity(NeuronPolarity.POS)}
                    >
                      Positive
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onSelect={() => setPolarity(NeuronPolarity.NEG)}
                    >
                      Negative
                    </DropdownMenuItem>
                    <DropdownMenuItem onSelect={() => setPolarity(null)}>
                      All
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div> */}
              <div className="grid grid-cols-4 items-center gap-4">
                <Label
                  htmlFor="top_k"
                  className="text-right flex items-center justify-end"
                >
                  Number of neurons
                  <Tooltip>
                    <TooltipTrigger tabIndex={-1}>
                      <HelpCircle
                        size={14}
                        className="ml-2 text-muted-foreground cursor-help"
                      />
                    </TooltipTrigger>
                    <TooltipContent className="text-sm text-center max-w-[20rem]">
                      Limit the number of neurons returned. Minimum is 1,
                      maximum is 1000.
                    </TooltipContent>
                  </Tooltip>
                </Label>
                <Input
                  id="top_k"
                  type="number"
                  className="col-span-3"
                  value={topK ?? undefined}
                  onChange={(e) => {
                    const value =
                      e.target.value === '' ? null : Number(e.target.value);
                    if (value === null || value <= MAX_TOP_K) {
                      setTopK(value);
                    } else {
                      setTopK(MAX_TOP_K);
                    }
                  }}
                  min={1}
                  max={MAX_TOP_K}
                  placeholder="Optional"
                />
              </div>
              {/* <div className="grid grid-cols-4 items-center gap-4">
                <Label
                  htmlFor="layer_range"
                  className="text-right flex items-center justify-end"
                >
                  Layer Range

                    <Tooltip>
                      <TooltipTrigger>
                        <HelpCircle
                          size={14}
                          className="ml-2 text-muted-foreground cursor-help"
                        />
                      </TooltipTrigger>
                      <TooltipContent className="text-sm text-center max-w-[20rem]">
                        Specify a range of layers to filter neurons. Leave empty
                        for all layers.
                      </TooltipContent>
                    </Tooltip>

                </Label>
                <div className="col-span-3 grid grid-cols-2 gap-2">
                  <Input
                    id="layer_range_start"
                    type="number"
                    placeholder="Start"
                    value={layerRange[0] ?? ''}
                    onChange={(e) =>
                      setLayerRange([
                        e.target.value ? Number(e.target.value) : null,
                        layerRange[1],
                      ])
                    }
                  />
                  <Input
                    id="layer_range_end"
                    type="number"
                    placeholder="End"
                    value={layerRange[1] ?? ''}
                    onChange={(e) =>
                      setLayerRange([
                        layerRange[0],
                        e.target.value ? Number(e.target.value) : null,
                      ])
                    }
                  />
                </div>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label
                  htmlFor="neuron_range"
                  className="text-right flex items-center justify-end"
                >
                  Neuron Range

                    <Tooltip>
                      <TooltipTrigger>
                        <HelpCircle
                          size={14}
                          className="ml-2 text-muted-foreground cursor-help"
                        />
                      </TooltipTrigger>
                      <TooltipContent className="text-sm text-center max-w-[20rem]">
                        Specify a range of neuron indices to filter. Leave empty
                        for all neurons.
                      </TooltipContent>
                    </Tooltip>

                </Label>
                <div className="col-span-3 grid grid-cols-2 gap-2">
                  <Input
                    id="neuron_range_start"
                    type="number"
                    placeholder="Start"
                    value={neuronRange[0] ?? ''}
                    onChange={(e) =>
                      setNeuronRange([
                        e.target.value ? Number(e.target.value) : null,
                        neuronRange[1],
                      ])
                    }
                  />
                  <Input
                    id="neuron_range_end"
                    type="number"
                    placeholder="End"
                    value={neuronRange[1] ?? ''}
                    onChange={(e) =>
                      setNeuronRange([
                        neuronRange[0],
                        e.target.value ? Number(e.target.value) : null,
                      ])
                    }
                  />
                </div>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label
                  htmlFor="explanation_score_range"
                  className="text-right flex items-center justify-end"
                >
                  Explanation Score Range

                    <Tooltip>
                      <TooltipTrigger>
                        <HelpCircle
                          size={14}
                          className="ml-2 text-muted-foreground cursor-help"
                        />
                      </TooltipTrigger>
                      <TooltipContent className="text-sm text-center max-w-[20rem]">
                        Filter neurons based on their explanation scores. Leave
                        empty for all scores.
                      </TooltipContent>
                    </Tooltip>

                </Label>
                <div className="col-span-3 grid grid-cols-2 gap-2">
                  <Input
                    id="explanation_score_range_start"
                    type="number"
                    placeholder="Start"
                    value={explanationScoreRange[0] ?? ''}
                    onChange={(e) =>
                      setExplanationScoreRange([
                        e.target.value ? Number(e.target.value) : null,
                        explanationScoreRange[1],
                      ])
                    }
                  />
                  <Input
                    id="explanation_score_range_end"
                    type="number"
                    placeholder="End"
                    value={explanationScoreRange[1] ?? ''}
                    onChange={(e) =>
                      setExplanationScoreRange([
                        explanationScoreRange[0],
                        e.target.value ? Number(e.target.value) : null,
                      ])
                    }
                  />
                </div>
              </div> */}
            </div>
            <div className="space-y-2">
              <div>
                <div className="text-md font-medium">Steering Mode</div>
                <div className="text-sm text-muted-foreground">
                  Whether to increase or zero out neuron activations.
                </div>
              </div>
              <div className="w-full flex">
                <Select
                  value={strength === 0 ? 'suppress' : 'strengthen'}
                  onValueChange={(value) =>
                    setStrength(value === 'suppress' ? 0 : DEFAULT_STRENGTH)
                  }
                >
                  <SelectTrigger
                    className={cn(
                      'mx-1 flex-grow',
                      strength === 0
                        ? 'text-red-600 hover:text-red-600'
                        : 'text-green-600 hover:text-green-600'
                    )}
                  >
                    <SelectValue placeholder="Select steering mode" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="strengthen">
                      <b>Strengthen</b>: set all activations to a high value
                    </SelectItem>
                    <SelectItem value="suppress">
                      <b>Suppress</b>: set all activations to 0
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
              {strength !== 0 && (
                <>
                  <div className="text-sm text-muted-foreground">
                    Setting activations to <b>{strength}x</b> the top quantile.
                  </div>
                  <div>
                    <Slider
                      min={0}
                      max={MAX_STRENGTH}
                      step={0.01}
                      value={[strength]}
                      onValueChange={(value) => setStrength(value[0])}
                    />
                  </div>
                </>
              )}
            </div>
            <div className="space-y-2 overflow-y-auto flex flex-col">
              <div>
                <div className="text-md font-medium">Steering Tokens</div>
                <div className="text-sm text-muted-foreground">
                  Neurons will be steered at selected tokens. Adjust the
                  existing selection by clicking and dragging.
                </div>
              </div>
              <div className="text-sm overflow-y-auto" ref={tokenSelectorRef}>
                <TokenSelector
                  chatTokens={chatTokens ?? []}
                  allowMultipleSelections={true}
                  initialSelection={selectedTokenRanges ?? undefined}
                  onSelectionChange={setSelectedTokenRanges}
                  alwaysTranslucent={true}
                  customSelectionClassName="bg-blue-200"
                  ignoreFirstToken={true}
                  disableUserSelection={isLoadingChat}
                />
              </div>
            </div>
          </ResizablePanel>
          <ResizableHandle withHandle className="mx-4" />
          <ResizablePanel minSize={40} className="flex flex-col space-y-2">
            <div className="text-md font-medium">
              Matching Neurons{' '}
              <span className="text-sm text-muted-foreground font-normal">
                ({neurons?.length})
              </span>
            </div>
            <div className="overflow-y-auto flex-grow">
              {neurons && (
                <DataTable
                  columns={columnsNoActivationSingleExplanation}
                  data={neurons?.map((neuron) =>
                    (() => {
                      let description = undefined;
                      let score = undefined;
                      if (neuron.polarity !== null) {
                        const descriptionObj =
                          neuronsMetadataDict?.general[
                            `${neuron.layer},${neuron.neuron}`
                          ]?.descriptions[neuron.polarity];
                        description = descriptionObj?.text;
                        score = descriptionObj?.score;
                      }

                      return {
                        ...neuron,
                        posDescription:
                          neuron.polarity === NeuronPolarity.POS
                            ? description
                            : undefined,
                        negDescription:
                          neuron.polarity === NeuronPolarity.NEG
                            ? description
                            : undefined,
                        score: score,
                      };
                    })()
                  )}
                  isLoading={isFetchingNeurons}
                  sortingState={[{ id: 'score', desc: true }]}
                />
              )}
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
        <DialogFooter>
          <Button type="submit" className="w-full" onClick={handleSubmit}>
            Steer with these neurons!
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
