import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { useEffect, useMemo, useState } from 'react';
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
import { NeuronDBFilter } from '@/app/types/neuronFilters';
import { PRESET_FLOWS } from '@/app/types/presetFlow';
import {
  setFlowState,
  setSteeringDialogSpec,
} from '@/app/store/slices/uiStateSlice';
import { cn } from '@/lib/utils';
import SteeringSpecDialog from './SteeringDialog';
import { usePostHog } from 'posthog-js/react';

const DEFAULT_STRENGTHENING_TOP_K = 50;
const DEFAULT_DEACTIVATING_TOP_K = 500;
const MAX_TOP_K = 1000;
const DEFAULT_STRENGTH = 0.75;
const MAX_STRENGTH = 1.0;

export default function SteeringPanel() {
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

  /**
   * Local state
   */
  const [concept, setConcept] = useState('');

  const posthog = usePostHog();

  /**
   * UI event handlers
   */
  const handleSteering = (version: 'add' | 'remove') => {
    if (concept == '') {
      toast({
        title: 'No concept provided',
        description: 'Please enter a concept to steer.',
      });
      return;
    }

    // PostHog tracking
    posthog.capture('Steering concept', {
      action: version,
      concept: concept,
      sessionId: sessionId,
    });

    dispatch(
      setSteeringDialogSpec({
        id: uuidv4(),
        name: concept,
        filter: {
          type: 'db',
          concept_or_embedding: concept,
          keyword: null,
          polarity: null,
          top_k:
            version == 'add'
              ? DEFAULT_STRENGTHENING_TOP_K
              : DEFAULT_DEACTIVATING_TOP_K,
          layer_range: [null, null],
          neuron_range: [null, null],
          explanation_score_range: [null, null],
        },
        tokenRanges: [
          [
            chatTokens
              ? (() => {
                  const reversedIndex = [...chatTokens]
                    .reverse()
                    .findIndex(
                      (token, index, array) =>
                        token.token === '<|start_header_id|>' &&
                        array
                          .slice(0, index)
                          .some((t) => t.token === '<|start_header_id|>')
                    );
                  return reversedIndex !== -1
                    ? chatTokens.length - 1 - reversedIndex
                    : 0;
                })()
              : 0,
            chatTokens
              ? chatTokens.findLastIndex(
                  (token) => token.token === '<|end_header_id|>'
                )
              : 0,
          ],
        ],
        strength: version == 'add' ? DEFAULT_STRENGTH : 0,
        isSteering: true,
      })
    );
    setConcept('');
  };

  /**
   * Preset flow
   */
  const flowState = useSelector((state: RootState) => state.uiState.flowState);
  const presetFlow = useMemo(() => {
    if (flowState?.presetFlowId) {
      return PRESET_FLOWS[flowState.presetFlowId];
    }
    return undefined;
  }, [flowState]);

  return (
    <div className="space-y-4 p-4 bg-white rounded-lg">
      <div className="space-y-2">
        <div>
          <div className="flex items-center">
            <div className="text-lg font-bold mr-2">Steering</div>

            {/* <Tooltip>
              <TooltipTrigger>
                <HelpCircle
                  size={14}
                  className="text-muted-foreground cursor-help"
                />
              </TooltipTrigger>
              <TooltipContent className="text-sm max-w-[35rem]">
                <ul className="list-disc pl-4">
                  <li>
                    You can directly manipulate neuron firings to change how the
                    model thinks.
                  </li>
                  <li>
                    Try strengthening or deactivating neurons corresponding to
                    some concepts.
                  </li>
                </ul>
              </TooltipContent>
            </Tooltip> */}
          </div>
          <div className="text-sm text-muted-foreground">
            Add or remove concepts from the model&apos;s computation.
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Input
            type="text"
            placeholder="Type a concept to steer with..."
            value={concept}
            onChange={(e) => setConcept(e.target.value)}
            className="h-8"
          />
          <Button
            onClick={() => handleSteering('add')}
            disabled={isLoadingChat || concept === ''}
            tabIndex={0}
            size="sm"
          >
            Strengthen
          </Button>
          <Button
            variant="outline"
            onClick={() => handleSteering('remove')}
            disabled={isLoadingChat || concept === ''}
            tabIndex={0}
            size="sm"
          >
            Suppress
          </Button>
        </div>
      </div>
      {steeringSpecs && (
        <div className="space-y-2 max-h-[20vh] overflow-y-auto">
          {steeringSpecs.map((spec) => (
            <SteeringSpecCard
              key={spec.id}
              steeringSpecId={spec.id}
              onEdit={() => {
                dispatch(setSteeringDialogSpec(spec));
              }}
            />
          ))}
        </div>
      )}
      <SteeringSpecDialog />
    </div>
  );
}

interface SteeringSpecCardProps {
  steeringSpecId: string;
  onEdit: () => void;
}

function SteeringSpecCard({ steeringSpecId, onEdit }: SteeringSpecCardProps) {
  /**
   * Global state
   */
  const dispatch = useDispatch();
  const sessionId = useSelector((state: RootState) => state.uiState.sessionId);
  const posthog = usePostHog();
  const steeringSpecs = useSelector(
    (state: RootState) => state.steering.steeringSpecs
  );
  const steeringSpec = steeringSpecs?.find(
    (spec) => spec.id === steeringSpecId
  );
  const isStrengthening = useMemo(
    () => steeringSpec?.strength ?? 0 > 0,
    [steeringSpec?.strength]
  );
  const isLoadingChat = useSelector(
    (state: RootState) => state.chat.isLoadingChat
  );

  if (steeringSpec == undefined) {
    return null;
  }

  /**
   * UI event handlers
   */
  const handleDelete = () => {
    posthog.capture('Deleted steering spec', {
      specId: steeringSpecId,
      sessionId: sessionId,
    });
    dispatch(removeSteeringSpec(steeringSpecId));
  };

  const handleToggleChange = (checked: boolean) => {
    posthog.capture('Toggled steering spec', {
      specId: steeringSpecId,
      enabled: checked,
      sessionId: sessionId,
    });

    dispatch(
      setSteeringSpec({
        id: steeringSpecId,
        spec: { ...steeringSpec, isSteering: checked },
      })
    );
  };

  return (
    <Card
      className={cn(
        'p-2 rounded-lg space-y-1',
        steeringSpec.isSteering &&
          (isStrengthening
            ? 'border-2 border-green-500'
            : 'border-2 border-red-500'),
        isLoadingChat && 'opacity-50 pointer-events-none'
      )}
    >
      <div className="flex flex-row items-center justify-between">
        <div className="flex items-center space-x-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <div
                className={`text-xs px-2 py-1 rounded-md cursor-default ${
                  steeringSpec.isSteering
                    ? isStrengthening
                      ? 'bg-green-100 text-green-800'
                      : 'bg-red-100 text-red-800'
                    : 'bg-gray-100 text-gray-500'
                }`}
              >
                {isStrengthening
                  ? // ? `Strengthening ${steeringSpec.filter.top_k} neurons`
                    // : `Suppressing ${steeringSpec.filter.top_k} neurons`}
                    `Strengthening`
                  : `Suppressing`}
              </div>
            </TooltipTrigger>
            <TooltipContent>
              {isStrengthening
                ? "Strengthens this concept's influence on the model"
                : "Removing this concept from the model's computation"}
            </TooltipContent>
          </Tooltip>

          <div className="text-md font-semibold line-clamp-1">
            {steeringSpec.filter.concept_or_embedding &&
              steeringSpec.filter.concept_or_embedding}
          </div>
        </div>
      </div>
      <div className="flex flex-wrap items-center space-x-2">
        <div className="flex items-center text-xs space-x-1">
          <div>Disable</div>
          <Switch
            checked={steeringSpec.isSteering}
            onCheckedChange={handleToggleChange}
          />
          <div>Enable</div>
        </div>
        <div className="flex-1" />
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            // PostHog tracking
            posthog.capture('Opened edit neurons dialog', {
              specId: steeringSpecId,
              sessionId: sessionId,
            });
            onEdit();
          }}
        >
          <EditIcon className="h-3 w-3 mr-1" /> Edit neurons
        </Button>
        <Button variant="outline" size="sm" onClick={handleDelete}>
          <TrashIcon className="h-3 w-3 mr-1" /> Delete
        </Button>
      </div>
    </Card>
  );
}
