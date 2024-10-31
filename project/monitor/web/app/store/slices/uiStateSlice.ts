import { SteeringSpec } from '@/app/types/neuronFilters';
import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface FlowState {
  presetFlowId: string;

  showTokenHighlight?: boolean; // Undefined means never shown, false means shown and then hidden
  tokensSelected?: boolean;

  isDone?: boolean;
}

interface UIState {
  sessionId?: string;
  flowState?: FlowState;

  // Showing panels
  showChatArea: boolean;
  showNeuronsPanel: boolean;
  showSteeringPanel: boolean;
  steeringDialogSpec?: SteeringSpec | null; // null means no dialog is open, undefined means dialog is open but no specific spec is selected
  showLinterPanel: boolean;
}

const initialState: UIState = {
  sessionId: undefined,
  flowState: undefined,
  showChatArea: false,
  showNeuronsPanel: false,
  showSteeringPanel: false,
  steeringDialogSpec: null,
  showLinterPanel: false,
};

const uiStateSlice = createSlice({
  name: 'uiState',
  initialState,
  reducers: {
    resetUIState(state) {
      Object.assign(state, initialState);
    },
    setSessionId(state, action: PayloadAction<string>) {
      state.sessionId = action.payload;
    },
    setFlowState(state, action: PayloadAction<FlowState | undefined>) {
      state.flowState = action.payload;
    },
    setShowChatArea(state, action: PayloadAction<boolean>) {
      state.showChatArea = action.payload;
    },
    setShowNeuronsPanel(state, action: PayloadAction<boolean>) {
      state.showNeuronsPanel = action.payload;
    },
    setShowSteeringPanel(state, action: PayloadAction<boolean>) {
      state.showSteeringPanel = action.payload;
    },
    setSteeringDialogSpec(state, action: PayloadAction<SteeringSpec | null | undefined>) {
      state.steeringDialogSpec = action.payload;
    },
    setShowLinterPanel(state, action: PayloadAction<boolean>) {
      state.showLinterPanel = action.payload;
    },
  },
});

export const {
  resetUIState,
  setSessionId,
  setFlowState,
  setShowChatArea,
  setShowNeuronsPanel,
  setShowSteeringPanel,
  setSteeringDialogSpec,
  setShowLinterPanel,
} = uiStateSlice.actions;
export default uiStateSlice.reducer;
