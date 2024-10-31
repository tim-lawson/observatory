import { ChatToken } from '@/app/types/tokens';
import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface GenerationParameters {
  maxNewTokens: number;
  temperature: number;
}

interface ChatState {
  tokens?: ChatToken[];
  isLoadingChat: boolean; // Either streaming tokens or registering intervention
  isStreamingTokens: boolean; // For streaming tokens only
  generationParameters: GenerationParameters;
}

const initialState: ChatState = {
  tokens: undefined,
  isLoadingChat: false,
  isStreamingTokens: false,
  generationParameters: {
    maxNewTokens: 64,
    temperature: 0.2,
  },
};

const chatSlice = createSlice({
  name: 'chat',
  initialState,
  reducers: {
    resetChatState(state) {
      Object.assign(state, initialState);
    },
    setChatTokens(state, action: PayloadAction<ChatToken[] | undefined>) {
      state.tokens = action.payload;
    },
    setIsLoadingChat(state, action: PayloadAction<boolean>) {
      state.isLoadingChat = action.payload;
    },
    setIsStreamingTokens(state, action: PayloadAction<boolean>) {
      state.isStreamingTokens = action.payload;
    },
    setMaxNewTokens(state, action: PayloadAction<number>) {
      state.generationParameters.maxNewTokens = action.payload;
    },
    setTemperature(state, action: PayloadAction<number>) {
      state.generationParameters.temperature = action.payload;
    },
  },
});

export const { resetChatState, setChatTokens, setIsLoadingChat, setIsStreamingTokens, setMaxNewTokens, setTemperature } = chatSlice.actions;
export default chatSlice.reducer;
