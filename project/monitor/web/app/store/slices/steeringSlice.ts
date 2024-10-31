import { SteeringSpec } from '@/app/types/neuronFilters';
import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface SteeringState {
  steeringSpecs?: SteeringSpec[];
}

const initialState: SteeringState = {
  steeringSpecs: undefined,
};

const steeringSlice = createSlice({
  name: 'steering',
  initialState,
  reducers: {
    resetSteeringState(state) {
      Object.assign(state, initialState);
    },
    addSteeringSpec(state, action: PayloadAction<SteeringSpec>) {
      if (state.steeringSpecs === undefined) {
        state.steeringSpecs = [];
      }
      state.steeringSpecs.push(action.payload);
    },
    removeSteeringSpec(state, action: PayloadAction<string>) {
      if (state.steeringSpecs === undefined) {
        return;
      }
      state.steeringSpecs = state.steeringSpecs.filter(
        (spec) => spec.id !== action.payload
      );
    },
    resetSteeringSpecs(state) {
      state.steeringSpecs = undefined;
    },
    setSteeringSpec(state, action: PayloadAction<{ id: string; spec: SteeringSpec }>) {
      if (state.steeringSpecs === undefined) {
        return;
      }
      const { id, spec } = action.payload;
      const index = state.steeringSpecs.findIndex(s => s.id === id);
      if (index !== -1) {
        state.steeringSpecs[index] = spec;
      }
    },
  },
});

export const {
  resetSteeringState,
  addSteeringSpec,
  removeSteeringSpec,
  resetSteeringSpecs,
  setSteeringSpec,
} = steeringSlice.actions;
export default steeringSlice.reducer;
