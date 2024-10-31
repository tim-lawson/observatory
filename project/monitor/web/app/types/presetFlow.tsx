import { v4 as uuidv4 } from 'uuid';
import { SteeringSpec } from './neuronFilters';

export interface PresetFlowTokenHighlight {
  indexRange: [number, number];
  message: string | JSX.Element;
}

export interface PresetFlow {
  id: string;
  header: string;
  title: string;
  prompt: string;
  overrideMaxNewTokens?: number;
  seed_response?: string;
  initialShowNeuronsFrom: 'activation' | 'attribution';
  tokenHighlight: PresetFlowTokenHighlight;
  llmAsk: string;
  solutionQuestion: string;
  solutionAnswer: string;
  steeringNarration: string;
  endMessages: string[];
  nextFlowId?: string;
  solutionSteeringSpecs: SteeringSpec[];
}

const COMPARISON_911_FLOW: PresetFlow = {
  id: 'comparison_911',
  header: '9.9 < 9.11',
  title: "Fix AI's belief that 9.9 < 9.11",
  prompt: 'Which is bigger: 9.9 or 9.11?',
  seed_response: '9.11 is bigger than 9.8.',
  initialShowNeuronsFrom: 'attribution',
  tokenHighlight: {
    indexRange: [53, 53],
    message: (
      <>
        This token looks wrong! 9.11 is <i>less than</i> 9.9.
        <br />
        Click the token to surface neurons that influenced the mistake.
      </>
    ),
  },
  llmAsk: 'to compare 9.11 and 9.9',
  solutionQuestion:
    "Do you notice any unexpected patterns that might confuse the model? When you're ready to fix them, let me know!",
  solutionAnswer:
    'Both bible verses and calendar dates follow numbering systems where 9.11 comes after 9.9. It appears that the model mistakenly associated 9.11 with the September 11, 2001 terrorist attacks.',
  steeringNarration:
    "You can manipulate neuron activations to steer the model output. Let's suppress the influence of unrelated neurons by setting their activations to 0.",
  endMessages: [
    'Great! Try re-generating and checking token probabilities to make sure the effect is robust.',
    "We seem to have suppressed this bug by suppressing less than 0.2% of MLP neurons. But are our findings general? Let's try a harder example now.",
  ],
  nextFlowId: 'sort_numbers',
  solutionSteeringSpecs: [
    {
      id: uuidv4(),
      name: 'biblical verses',
      filter: {
        type: 'db',
        concept_or_embedding: 'biblical verses',
        keyword: null,
        top_k: 500,
      },
      tokenRanges: [[0, 47]],
      isSteering: true,
      strength: 0,
    },
    {
      id: uuidv4(),
      name: 'months and dates',
      filter: {
        type: 'db',
        concept_or_embedding: 'months and dates',
        keyword: null,
        top_k: 500,
      },
      tokenRanges: [[0, 47]],
      isSteering: true,
      strength: 0,
    },
  ],
};

const SORT_FLOW: PresetFlow = {
  id: 'sort_numbers',
  header: 'Sorting decimals',
  title: "Fix AI's inability to sort numbers",
  prompt:
    'Sort these decimal numbers from smallest to largest: 9.6, 9.7, 9.8, 9.9, 9.10, 9.11, 9.12, 9.13. They are NOT software versions!',
  seed_response: 'N/A',
  initialShowNeuronsFrom: 'attribution',
  tokenHighlight: {
    indexRange: [110, 130],
    message: (
      <>
        Something looks wrong! 9.10 should be the smallest number.
        <br />
        Click the token you think is wrong to surface neurons that influenced
        the mistake.
      </>
    ),
  },
  llmAsk: 'to sort some numbers.',
  solutionQuestion:
    "Try doing attribution to some more tokens. Do you notice any unexpected patterns that might confuse the model? When you're ready to fix them, let me know!",
  solutionAnswer:
    'The model seems confused by the numbering system for bible verses and calendar dates, both of which have 9.6 (e.g., September 6th) come before 9.10.',
  steeringNarration:
    "You can manipulate neuron activations to steer the model output. Let's suppress the influence of unrelated neurons by setting their activations to 0.",
  endMessages: [
    'It turns out that this dates & bible verses phenomenon is quite general. See https://transluce.org/observability-interface for details!',
    'Feel free to try out some of the other examples on the interface!',
  ],
  solutionSteeringSpecs: [
    {
      id: uuidv4(),
      name: 'dates and months',
      filter: {
        type: 'db',
        concept_or_embedding: 'dates and months',
        keyword: null,
        top_k: 500,
      },
      tokenRanges: [[0, 88]],
      isSteering: true,
      strength: 0,
    },
    {
      id: uuidv4(),
      name: 'biblical verses',
      filter: {
        type: 'db',
        concept_or_embedding: 'biblical verses',
        keyword: null,
        top_k: 500,
      },
      tokenRanges: [[0, 88]],
      isSteering: true,
      strength: 0,
    },
  ],
};

const CARLINI_FLOW: PresetFlow = {
  id: 'carlini_elk',
  header: 'Eliciting hidden knowledge',
  title: 'Elicit hidden knowledge',
  prompt: 'Who is Nicholas Carlini?',
  initialShowNeuronsFrom: 'activation',
  tokenHighlight: {
    indexRange: [30, 40],
    message: (
      <>
        Click and drag your mouse over these tokens to see which neurons fired
        while the model processed Nicholas&apos; name.
      </>
    ),
  },
  llmAsk: 'who Nicholas Carlini is',
  solutionQuestion:
    "Nicholas Carlini is a researcher in adversarial machine learning. He finds ways to fool models into undesirable behaviors. Do you see any evidence that the model knows this? Let me know when you're ready to try fixing the issue!",
  solutionAnswer:
    "Even though the model refused to answer, you can see cybersecurity concepts firing, which is relevant to Carlini's work.",
  steeringNarration:
    'You can manipulate neuron activations to steer the model output. Let\'s nudge the model in the direction of "machine learning" and see what it reveals.',
  endMessages: [
    'Interesting! The model seems to know more about Nicholas that it was originally letting on.',
    'Feel free to try out some of the other examples on the interface!',
  ],
  solutionSteeringSpecs: [
    {
      id: uuidv4(),
      name: 'machine learning',
      filter: {
        type: 'db',
        concept_or_embedding: 'machine learning',
        keyword: null,
        top_k: 75,
      },
      tokenRanges: [[30, 40]],
      isSteering: true,
      strength: 0.6,
    },
  ],
};

const ALICE_BOB_FLOW: PresetFlow = {
  id: 'alice_bob_steering',
  header: 'Multi-concept steering',
  title: 'Steer specific characters in a story',
  prompt: 'Tell me a quick story about Alice and Bob.',
  overrideMaxNewTokens: 128,
  initialShowNeuronsFrom: 'activation',
  tokenHighlight: {
    indexRange: [38, 38],
    message: (
      <>
        Click on Bob&apos;s name to see what the model was thinking at that
        point.
      </>
    ),
  },
  llmAsk: 'for a quick story',
  solutionQuestion:
    'It turns out that Alice and Bob are common characters in cryptography and quantum mechanics. The model knew this as soon as it saw Bob! Want to see how we can change the story up?',
  solutionAnswer: 'What if we try to make Alice a fish, and Bob a bird?',
  steeringNarration:
    'For fine-grained control, we\'ll choose specific tokens to manipulate neuron activations at. Let\'s strengthen "fish" neurons at Alice and "bird" neurons at Bob.',
  endMessages: [
    'Try re-generating a couple times to see different stories, or feel free to try out some of the other examples on the interface!',
  ],
  solutionSteeringSpecs: [
    {
      id: uuidv4(),
      name: 'bird',
      filter: {
        type: 'db',
        concept_or_embedding: 'bird',
        keyword: null,
        top_k: 75,
      },
      tokenRanges: [[36, 36]],
      isSteering: true,
      strength: 0.75,
    },
    {
      id: uuidv4(),
      name: 'fish',
      filter: {
        type: 'db',
        concept_or_embedding: 'fish',
        keyword: null,
        top_k: 75,
      },
      tokenRanges: [[38, 38]],
      isSteering: true,
      strength: 0.75,
    },
  ],
};

export const PRESET_FLOWS = {
  [COMPARISON_911_FLOW.id]: COMPARISON_911_FLOW,
  [SORT_FLOW.id]: SORT_FLOW,
  // [COMPARISON_912_FLOW.id]: COMPARISON_912_FLOW,
  [CARLINI_FLOW.id]: CARLINI_FLOW,
  [ALICE_BOB_FLOW.id]: ALICE_BOB_FLOW,
};

// export interface SteeringSpec {
//     id: string;
//     name: string;
//     filter: NeuronDBFilter;
//     tokenRanges: [number, number][];
//     strength: number;
//     isSteering?: boolean;
//   }

//   export interface NeuronDBFilter {
//     type: 'db';
//     concept_or_embedding: string | null;
//     keyword: string | null;
//     polarity?: NeuronPolarity | null;
//     top_k: number | null;
//     layer_range?: [number | null, number | null];
//     neuron_range?: [number | null, number | null];
//     explanation_score_range?: [number | null, number | null];
//   }
