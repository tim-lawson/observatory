import { useDispatch, useSelector } from 'react-redux';
import {
  DataTable,
  DEFAULT_ACTIVATION_SORTING_STATE,
  DEFAULT_ATTRIBUTION_SORTING_STATE,
} from './datatable';
import {
  setDescriptionKeywordFilter,
  setMousedOverNeurons,
  setSelectedAttributionToken,
  setSelectedTokenRange,
  setShowNeuronsFrom,
} from '@/app/store/slices/neuronsSlice';
import { columnsAttribution, columnsSingleExplanation } from './columns';
import { useEffect, useMemo, useState } from 'react';
import { RootState } from '@/app/store/store';
import { NeuronForDisplay, NeuronPolarity } from '@/app/types/neuronData';
import { Input } from '@/components/ui/input';
import { useDebounce } from '@/hooks/use-debounce';
import { X } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { HelpCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { PRESET_FLOWS } from '@/app/types/presetFlow';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Loader2 } from 'lucide-react';
import { InfoIcon } from 'lucide-react';
import { toast } from '@/hooks/use-toast';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { setSelectedClusterId } from '@/app/store/slices/aiLinterSlice';
import { usePostHog } from 'posthog-js/react';

export default function NeuronDisplay() {
  /**
   * Global state
   */
  const dispatch = useDispatch();
  const showNeuronsFrom = useSelector(
    (state: RootState) => state.neurons.displayModulation.showNeuronsFrom
  );
  const chatTokens = useSelector((state: RootState) => state.chat.tokens);
  const selectedTokenRange = useSelector(
    (state: RootState) => state.neurons.displayModulation.selectedTokenRange
  );
  const neuronsMetadataDict = useSelector(
    (state: RootState) => state.neurons.metadataDict
  );
  const descriptionKeywordFilter = useSelector(
    (state: RootState) =>
      state.neurons.displayModulation.descriptionKeywordFilter
  );
  const selectedAttributionToken = useSelector(
    (state: RootState) =>
      state.neurons.displayModulation.selectedAttributionToken
  );
  const tableHighlightedNeuronIds = useSelector(
    (state: RootState) =>
      state.neurons.displayModulation.tableHighlightedNeuronIds
  );
  const tableHighlightedNeuronIdsSet = useMemo(() => {
    if (tableHighlightedNeuronIds) {
      return new Set(tableHighlightedNeuronIds);
    }
    return undefined;
  }, [tableHighlightedNeuronIds]);

  /**
   * Local state
   */
  const [actAttrExplanationOpen, setActAttrExplanationOpen] = useState(false);
  const [localDescriptionKeywordFilter, setLocalDescriptionKeywordFilter] =
    useState<string | undefined>(undefined);
  const debouncedDescriptionKeywordFilter = useDebounce(
    localDescriptionKeywordFilter,
    100
  );
  useEffect(() => {
    dispatch(setDescriptionKeywordFilter(debouncedDescriptionKeywordFilter));
  }, [debouncedDescriptionKeywordFilter]);

  const activeFilters = useMemo(() => {
    const filters = [];
    if (descriptionKeywordFilter) {
      filters.push({
        name: `Keyword: ${descriptionKeywordFilter}`,
        onDelete: () => dispatch(setDescriptionKeywordFilter(undefined)),
      });
    }
    if (selectedTokenRange) {
      filters.push({
        name: `Tokens: [${selectedTokenRange[0]}, ${selectedTokenRange[1]}]`,
        onDelete: () => dispatch(setSelectedTokenRange(undefined)),
      });
    }
    return filters;
  }, [descriptionKeywordFilter, selectedTokenRange]);

  /**
   * Handle UI events
   */

  const posthog = usePostHog();
  const sessionId = useSelector((state: RootState) => state.uiState.sessionId);

  const handleResetAttributionToken = () => {
    posthog.capture('Reset attribution token', {
      sessionId: sessionId,
    });
    dispatch(setSelectedTokenRange(undefined));
    dispatch(setSelectedAttributionToken(undefined));
  };

  /**
   * Get neurons we should display
   */
  const neuronsForDisplay = useMemo(() => {
    const neuronGroups: { [key: string]: NeuronForDisplay[] } = {};

    const general = neuronsMetadataDict?.general;
    const run = neuronsMetadataDict?.run;

    if (general === undefined || run === undefined) {
      console.error('Neuron metadata not loaded properly');
      return [];
    }

    const ans = Object.values(run).map((cur) => {
      // Filter based on selected token range
      if (
        selectedTokenRange !== undefined &&
        (cur.token < selectedTokenRange[0] || cur.token > selectedTokenRange[1])
      ) {
        return;
      }

      const lnKey = `${cur.layer},${cur.neuron}`;

      const activation = cur.activation;
      const topQuantile = general[lnKey]?.activation_percentiles['0.99999'];
      const bottomQuantile = general[lnKey]?.activation_percentiles['1e-05'];

      // Determine polarity
      const polarity =
        topQuantile !== undefined && activation >= topQuantile - 1e-2
          ? NeuronPolarity.POS
          : NeuronPolarity.NEG;

      // Get metadata dependent on polarity
      const activationNormalized =
        activation /
        (polarity === NeuronPolarity.POS ? topQuantile : bottomQuantile);
      const isInteresting =
        general[lnKey]?.descriptions[polarity]?.is_interesting;
      const score = general[lnKey]?.descriptions[polarity]?.score;

      // Get descriptions
      const posDescription =
        general[lnKey]?.descriptions[NeuronPolarity.POS]?.text;
      const negDescription =
        general[lnKey]?.descriptions[NeuronPolarity.NEG]?.text;
      const description =
        polarity == NeuronPolarity.POS ? posDescription : negDescription;
      if (description === undefined) {
        return;
      }

      // Filter based on interestingness
      if (!isInteresting) {
        return;
      }

      // Filter based on description keyword filter
      if (
        descriptionKeywordFilter !== undefined &&
        !description
          .toLowerCase()
          .includes(descriptionKeywordFilter.toLowerCase())
      ) {
        return;
      }

      // Get attributions
      const attributions =
        run[`${cur.layer},${cur.neuron},${cur.token}`]?.attributions;
      const firstAttributionEntry = attributions
        ? Object.entries(attributions)[0]
        : undefined;
      const firstAttribution = firstAttributionEntry
        ? firstAttributionEntry[1]
        : undefined;

      if (!neuronGroups[lnKey]) {
        neuronGroups[lnKey] = [];
      }
      neuronGroups[lnKey].push({
        layer: cur.layer,
        neuron: cur.neuron,
        token: cur.token,
        polarity,
        activation,
        attribution: firstAttribution?.attribution,
        activationNormalized: activationNormalized,
        posDescription:
          polarity == NeuronPolarity.POS ? posDescription : undefined,
        negDescription:
          polarity == NeuronPolarity.NEG ? negDescription : undefined,
        isInteresting,
        score,
        inSelectedCluster: tableHighlightedNeuronIdsSet?.has(lnKey),
      });
    });

    // Return the neuron with the highest normalized activation for each neuron group
    return Object.keys(neuronGroups).map((key) =>
      neuronGroups[key].reduce((maxNeuron, currentNeuron) =>
        currentNeuron.activationNormalized && maxNeuron.activationNormalized
          ? currentNeuron.activationNormalized > maxNeuron.activationNormalized
            ? currentNeuron
            : maxNeuron
          : maxNeuron
      )
    );
  }, [
    neuronsMetadataDict,
    selectedTokenRange,
    descriptionKeywordFilter,
    tableHighlightedNeuronIdsSet,
  ]);

  const showTable = useMemo(() => {
    if (showNeuronsFrom === 'activation') {
      return true;
    } else if (selectedAttributionToken !== undefined) {
      return true;
    }
    return false;
  }, [showNeuronsFrom, selectedAttributionToken]);

  return (
    <>
      <div className="flex items-center space-x-4 mb-2">
        <Tabs
          value={
            showNeuronsFrom === 'activation' ? 'activation' : 'attribution'
          }
          onValueChange={(value) => {
            if (value === 'activation' || value === 'attribution') {
              posthog.capture('Changed neuron display mode', {
                mode: value,
                sessionId: sessionId,
              });
              dispatch(setShowNeuronsFrom(value));
            }
          }}
        >
          <div className="flex items-center">
            <TabsList className="w-full">
              <TabsTrigger value="activation" className="w-full">
                Activation Mode
              </TabsTrigger>
              <TabsTrigger value="attribution" className="w-full">
                Attribution Mode
              </TabsTrigger>
            </TabsList>
            <Tooltip>
              <TooltipTrigger asChild>
                {/* <Button
                  variant="secondary"
                  size="sm"
                  className="ml-1"
                  onClick={() => setActAttrExplanationOpen(true)}
                >
                  <HelpCircle
                    onClick={() => setActAttrExplanationOpen(true)}
                    className="h-5 w-5"
                  />
                </Button> */}
                <HelpCircle
                  onClick={() => setActAttrExplanationOpen(true)}
                  className="h-4 w-4 ml-2 cursor-pointer text-muted-foreground"
                />
              </TooltipTrigger>
              <TooltipContent>
                Click to learn about the difference between activation and
                attribution.
              </TooltipContent>
            </Tooltip>
          </div>
        </Tabs>
        <div>
          {showNeuronsFrom === 'activation' && (
            <>
              <div className="flex items-center space-x-2">
                <div className="text-lg font-bold">High-Activation Neurons</div>
              </div>
              <div className="text-sm text-muted-foreground">
                Showing neurons that fire highly on
                {selectedTokenRange === undefined ? (
                  ' your prompt.'
                ) : (
                  <div className="inline-flex items-center ml-2 bg-gray-200 text-secondary-foreground px-1.5 py-0.5 rounded-md text-xs shadow-sm border border-secondary-foreground/20">
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
                  </div>
                )}
              </div>
            </>
          )}
          {showNeuronsFrom === 'attribution' && (
            <>
              <div className="flex items-center space-x-2">
                <div className="text-lg font-bold">High-Influence Neurons</div>
              </div>
              <div className="text-sm text-muted-foreground">
                Showing neurons that promoted the prediction of:
                {selectedAttributionToken === undefined && (
                  <div className="inline ml-2 bg-gray-200 text-secondary-foreground px-1.5 py-0.5 rounded-md font-mono text-xs shadow-sm border border-secondary-foreground/20">
                    Select a token
                  </div>
                )}
                {selectedAttributionToken !== undefined && (
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
                      onClick={handleResetAttributionToken}
                    >
                      <X size={12} />
                    </Button>
                  </div>
                )}
                {selectedTokenRange && (
                  <>
                    {' '}
                    and fired at:
                    <div className="inline-flex items-center ml-2 bg-gray-200 text-secondary-foreground px-1.5 py-0.5 rounded-md text-xs shadow-sm border border-secondary-foreground/20">
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
                    </div>
                  </>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      {showTable && (
        <>
          <div className="space-y-2 mb-2">
            <Input
              placeholder="Quickly search neuron descriptions by keyword..."
              value={localDescriptionKeywordFilter ?? ''}
              onChange={(e) => {
                if (e.target.value === '') {
                  setLocalDescriptionKeywordFilter(undefined);
                } else {
                  setLocalDescriptionKeywordFilter(e.target.value);
                }
                posthog.capture('Searched neuron descriptions', {
                  keyword: e.target.value,
                  sessionId: sessionId,
                });
              }}
              className="flex-grow h-8"
            />
            {/* {activeFilters.length > 0 && (
              <div className="flex flex-wrap gap-2">
                <span className="ml-1 text-xs inline-flex items-center text-gray-600">
                  Active filters:
                </span>
                {activeFilters.map((filter, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center pr-0.5 pl-1.5 py-0.5 rounded-full text-xxs bg-primary/10 text-primary"
                  >
                    {filter.name}
                    <button
                      onClick={() => filter.onDelete()}
                      className="rounded-full p-0.5 hover:bg-primary/20 transition-colors"
                    >
                      <X size={12} />
                    </button>
                  </span>
                ))}
              </div>
            )} */}
          </div>

          <DataTable
            columns={
              showNeuronsFrom === 'activation'
                ? columnsSingleExplanation
                : columnsAttribution
            }
            sortingState={
              showNeuronsFrom === 'activation'
                ? DEFAULT_ACTIVATION_SORTING_STATE
                : DEFAULT_ATTRIBUTION_SORTING_STATE
            }
            data={neuronsForDisplay}
            onMouseOverRow={(neuron) =>
              dispatch(setMousedOverNeurons(neuron ? [neuron] : undefined))
            }
          />
        </>
      )}

      {!showTable && (
        <div className="flex flex-1 items-center justify-center border rounded-lg">
          <div className="text-center text-md text-muted-foreground">
            <InfoIcon className="mx-auto mb-2 h-6 w-6 text-muted-foreground" />
            Select a token in the chat to find neurons that contributed highly
            to its prediction.
          </div>
        </div>
      )}

      <Dialog
        open={actAttrExplanationOpen}
        onOpenChange={setActAttrExplanationOpen}
      >
        <DialogContent
          className="max-w-4xl text-base" // Add this line to set the maximum width
        >
          <DialogHeader>
            <DialogTitle className="text-xl">
              What&apos;s the difference between activation and attribution?
            </DialogTitle>
          </DialogHeader>
          <div>
            Activation and attribution are two different ways of determining
            which neurons are important in a given context.
          </div>
          <div>
            <b>Activation</b> measures the (normalized) activation value of the
            neuron. Llama uses gated MLPs, meaning that activations can be
            either positive or negative. We normalize by the value of the 10
            <sup>-5</sup> quantile of the neuron across a large dataset of
            examples.
          </div>
          <div>
            <b>Attribution</b> measures how much the neuron affects the model’s
            output. Attribution must be conditioned on a specific output token,
            and is equal to the gradient of that output token’s probability with
            respect to the neuron’s activation, times the activation value of
            the neuron. Attribution values are not normalized, and are reported
            as absolute values.
          </div>
          <div>
            We recommend starting with activations to build an initial sense of
            the model’s representations, then drilling down in one of two ways:
            <ul className="list-disc pl-8">
              <li>
                If you want to understand why a specific behavior occurs, apply
                attribution at the token most relevant to that behavior.
              </li>
              <li>
                If you want to uncover hidden knowledge, look for tokens that
                activate interesting features (these are highlighted in red or
                green), then stay in activation mode and click those tokens to
                see what other features are firing.
              </li>
            </ul>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
