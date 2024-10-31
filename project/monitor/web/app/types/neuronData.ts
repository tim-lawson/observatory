export enum NeuronPolarity {
  POS = '1',
  NEG = '-1',
}

export interface NeuronWithDescription extends Neuron {
  description: string;
}

export interface NeuronCluster {
  id: string;
  neurons: NeuronWithDescription[];
  description: string;
  similarity: number;
  isSelected?: boolean;
}

export interface NeuronDescription {
  text: string;
  score?: number;
  is_interesting?: boolean;
}

export interface Neuron {
  layer: number;
  neuron: number;
  token: number | null;
  polarity: NeuronPolarity | null;
}

export interface NeuronForDisplay extends Neuron {
  activation?: number;
  activationNormalized?: number;
  attribution?: number;
  posDescription?: string;
  negDescription?: string;
  score?: number;
  isInteresting?: boolean;
  inSelectedCluster?: boolean;
}

export interface NeuronGeneralMetadata {
  layer: number;
  neuron: number;
  descriptions: {
    [key in NeuronPolarity]: NeuronDescription;
  };
  activation_percentiles: {
    [key: string]: number;
  };
}

export interface NeuronRunMetadata {
  layer: number;
  neuron: number;
  token: number;
  activation: number;
  attributions?: { [key: number]: AttributionResult };
}

export interface NeuronsMetadataDict {
  general: { [key: string]: NeuronGeneralMetadata }; // key is `${neuron.layer},${neuron.neuron}`
  run: { [key: string]: NeuronRunMetadata }; // key is `${neuron.layer},${neuron.neuron},${neuron.token}`
}

export interface AttributionResult {
  src_token_idx: number;
  tgt_token_idx: number;
  attribution: number;
}
