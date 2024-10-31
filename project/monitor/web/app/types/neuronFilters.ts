import { Neuron, NeuronPolarity } from "./neuronData";

export type NeuronFilter =
  | NeuronDBFilter
  | TokenFilter
  | IdsFilter
  | ActivationPercentileFilter
  | DescriptionKeywordFilter
  | ComplexFilter
  | AttributionFilter;

export interface SteeringSpec {
  id: string;
  name: string;
  filter: NeuronDBFilter;
  tokenRanges: [number, number][];
  strength: number;
  isSteering?: boolean;
}

export interface NeuronDBFilter {
  type: 'db';
  concept_or_embedding: string | null;
  keyword: string | null;
  polarity?: NeuronPolarity | null;
  top_k: number; // Disallow null, since that'd overload the server
  layer_range?: [number | null, number | null];
  neuron_range?: [number | null, number | null];
  explanation_score_range?: [number | null, number | null];
}

export interface AttributionFilter {
  type: 'attribution';
  target_token_idx: number;
  top_k: number;
}

export interface ActivationPercentileFilter {
  type: 'activation_percentile';
  percentile: '1e-6' | '1e-5' | '1e-4' | '1e-3' | '1e-2';
  direction: 'top' | 'bottom';
}

export interface ComplexFilter {
  type: 'complex';
  filters: NeuronFilter[];
  op: 'and' | 'or';
}

export interface TokenFilter {
  type: 'token';
  tokens: number[]
}

export interface LocalTokenFilter {
  type: 'token';
  startIdx: number;
  endIdx: number;
}

export interface IdsFilter {
  type: 'ids';
  ids: Neuron[];
}

export interface DescriptionKeywordFilter {
  type: 'description_keyword';
  description: string;
}
