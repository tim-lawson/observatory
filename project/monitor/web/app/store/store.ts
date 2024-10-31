// store.ts

import { configureStore } from '@reduxjs/toolkit';
import neuronsReducer from './slices/neuronsSlice';
import chatReducer from './slices/chatSlice';
import aiLinterReducer from './slices/aiLinterSlice';
import steeringReducer from './slices/steeringSlice';
import uiStateReducer from './slices/uiStateSlice';
import { neuronsApi } from './api/neuronsApi';
import { sessionApi } from './api/sessionApi';
import { sendMessageApi, chatApi } from './api/chatApi';

const store = configureStore({
  reducer: {
    neurons: neuronsReducer,
    chat: chatReducer,
    aiLinter: aiLinterReducer,
    steering: steeringReducer,
    uiState: uiStateReducer,
    [neuronsApi.reducerPath]: neuronsApi.reducer,
    [sessionApi.reducerPath]: sessionApi.reducer,
    [sendMessageApi.reducerPath]: sendMessageApi.reducer,
    [chatApi.reducerPath]: chatApi.reducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(neuronsApi.middleware, sessionApi.middleware, sendMessageApi.middleware, chatApi.middleware),
});


export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

export default store;
