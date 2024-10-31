import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import { BASE_URL } from '@/app/constants';

interface FetchNeuronsParams {
  sessionId: string;
  activationQuantileThreshold: string;
}

export const sessionApi = createApi({
  reducerPath: 'sessionApi',
  baseQuery: fetchBaseQuery({ baseUrl: BASE_URL }), // Adjust the base URL as needed
  endpoints: (builder) => ({
    fetchSession: builder.query<
      string,
      void
    >({
      query: () => ({
        url: `register`,
        method: 'POST',
      }),
    }),
  }),
});

export const { useFetchSessionQuery } = sessionApi;
