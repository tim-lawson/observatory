import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Neuron, NeuronsMetadataDict } from '@/app/types/neuronData';
import { ComplexFilter, NeuronFilter } from '@/app/types/neuronFilters';

export const DEFAULT_ACTIVATION_FILTER = {
  type: 'complex',
  op: 'or',
  filters: [
    {
      type: 'activation_percentile',
      percentile: '1e-5',
      direction: 'bottom',
    },
    {
      type: 'activation_percentile',
      percentile: '1e-5',
      direction: 'top',
    },
  ],
} as ComplexFilter;

export interface NeuronDisplayModulation {
  selectedTokenRange?: [number, number];
  mousedOverNeurons?: Neuron[];
  mousedOverTokenIndex?: number;
  selectedAttributionToken?: number;
  showNeuronsFrom: 'attribution' | 'activation';
  descriptionKeywordFilter?: string;
  tableHighlightedNeuronIds?: string[];
}

interface NeuronsState {
  globalNeuronFilter?: NeuronFilter;
  metadataDict?: NeuronsMetadataDict;
  displayModulation: NeuronDisplayModulation;
  activationQuantileThreshold: string,
}

const initialState: NeuronsState = {
  globalNeuronFilter: DEFAULT_ACTIVATION_FILTER,
  metadataDict: undefined,
  displayModulation: {
    selectedTokenRange: undefined,
    selectedAttributionToken: undefined,
    mousedOverNeurons: undefined,
    mousedOverTokenIndex: undefined,
    showNeuronsFrom: 'activation',
    descriptionKeywordFilter: undefined,
    tableHighlightedNeuronIds: undefined,
  },
  activationQuantileThreshold: '1e-5',
};

const neuronsSlice = createSlice({
  name: 'neurons',
  initialState,
  reducers: {
    resetNeuronsState(state) {
      Object.assign(state, initialState);
    },
    setGlobalNeuronFilter(state, action: PayloadAction<NeuronFilter | undefined>) {
      state.globalNeuronFilter = action.payload;
    },
    setNeuronsMetadataDict(state, action: PayloadAction<NeuronsMetadataDict>) {
      state.metadataDict = action.payload;
    },
    setSelectedTokenRange(state, action: PayloadAction<[number, number] | undefined>) {
      state.displayModulation.selectedTokenRange = action.payload;
    },
    setMousedOverNeurons(state, action: PayloadAction<Neuron[] | undefined>) {
      state.displayModulation.mousedOverNeurons = action.payload;
    },
    setMousedOverTokenIndex(state, action: PayloadAction<number | undefined>) {
      state.displayModulation.mousedOverTokenIndex = action.payload;
    },
    setSelectedAttributionToken(state, action: PayloadAction<number | undefined>) {
      state.displayModulation.selectedAttributionToken = action.payload;
    },
    setShowNeuronsFrom(state, action: PayloadAction<'attribution' | 'activation'>) {
      state.displayModulation.showNeuronsFrom = action.payload;
    },
    setActivationQuantileThreshold(state, action: PayloadAction<string>) {
      state.activationQuantileThreshold = action.payload;
    },
    setDescriptionKeywordFilter(state, action: PayloadAction<string | undefined>) {
      state.displayModulation.descriptionKeywordFilter = action.payload;
    },
    setTableHighlightedNeuronIds(state, action: PayloadAction<string[] | undefined>) {
      state.displayModulation.tableHighlightedNeuronIds = action.payload;
    },
  },
});

export const {
  resetNeuronsState,
  setGlobalNeuronFilter,
  setNeuronsMetadataDict,
  setSelectedTokenRange,
  setMousedOverNeurons,
  setMousedOverTokenIndex,
  setSelectedAttributionToken,
  setShowNeuronsFrom,
  setActivationQuantileThreshold,
  setDescriptionKeywordFilter,
  setTableHighlightedNeuronIds,
} = neuronsSlice.actions;
export default neuronsSlice.reducer;
