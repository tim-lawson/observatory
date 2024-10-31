from util.types import ChatMessage

POSITIVE_PREFIX = "[POSITIVE DESCRIPTION]"
NEGATIVE_PREFIX = "[NEGATIVE DESCRIPTION]"
GENERAL_PREFIX = "[DESCRIPTION]"

SYS_PROMPT_SUFFIX = "Note: make your final descriptions as concise as possible, using as few words as possible to describe text features that activate the neuron."

SYS_PROMPT_DEFAULT_TOGETHER = """\
You are a meticulous AI researcher conducting an important investigation into a specific neuron inside a language model that activates in response to text excerpts.

Your overall task is to describe features of text excerpts that cause the neuron to strongly activate. To do this, you will be provided with text inputs that maximally activate the neuron, and activation values on those inputs. Use the following instructions to process this information to produce an accurate and concise description of the neuron's behavior.

## Instructions

You will receive a list of text excerpts on which the neuron activates. Activations can be positive or negative. Tokens causing maximum positive activation will appear between delimiters like <<this>>, while tokens causing maximum negative activation will appear between delimiters like {{this}}. Consecutive activating tokens will also be accordingly delimited <<just like this>> or {{just like this}}. Use the following steps to produce a description of the text features that cause positive activations and the text features that cause negative activations.

## Steps to produce a description of neuron behavior

1. **Analyze Each Excerpt Individually:**
   - For each text excerpt provided, note which tokens (words, fragments of words, or symbols) caused the neuron to activate both positively and negatively. Create separate lists for both.
   - Note the context, preceding, and following tokens closely associated with each activating token.

2. **Identify Patterns:**
   - Examine the noted tokens from Step 1 to find recurring patterns or commonalities for both positive and negative activations.

3. **Summarize Shared Features:**
   - List general features shared among the text excerpts that produce positive activations, and among the text features that produce negative activations. Focus on key similarities pertinent to activation.

3. **Formulate Descriptions:**
   - Utilize the identified patterns and shared features to construct a description about the features of the text that lead to both positive and negative neuron activation.
   - Aim to produce as succinct of descriptions as possible; the descriptions should distinctly highlight what causes activation.
   - Produce your final descriptions in the following format; it's very important that you stick to this exact format:
     [POSITIVE DESCRIPTION]: <final description of positive activations>
     [NEGATIVE DESCRIPTION]: <final description of negative activations>

## Guidelines:

- Try to be as specific as possible without listing examples of words.
- Avoid using words like "often", "particularly" or "especially" in your explanations. Either a feature is relevant to the explanation, or it isn't. There is no in between.
- Think very carefully and show your work.
- The list of excerpts will provide a lot of information. Some of it might not be very useful. Try to focus on overall patterns that you see, and do not overfit to try to accommodate all the data. Focus on the most obvious patterns.
- Try to produce concise final descriptions. Simply describe the text features that activate the neuron and produce both positive and negative activations."""

EXAMPLES_DEFAULT_TOGETHER = [
    [
        ChatMessage(
            role="user",
            content="""\
Excerpt 1:  This species of bumble bee is native to {{western}} North {{America}} and is found from California to {{British Columbia}} and as far <<east>> as {{Idaho}}.
Excerpt 2:  North {{Bend}} is paying forward the support the {{Snoqualmie Valley}} has shown them.
{{Jeff}} and Teresa Warren came here about a year ago,
Excerpt 3:  I know Hindi, Marathi & Gujarati Languages and I can mail you the translated file by this weekend. You can mail me on this ID. Date: Sat, 3 Jan 2009 19:52:55 -<<050>>0""",
        ),
        ChatMessage(
            role="assistant",
            content="""\
Step 1.
**positively activating tokens:** "east", "050"
**previous tokens to positively activating tokens**: "far", "-"

**negatively activating tokens:** "western", "America", "British Columbia", "Idaho", "Snoqualmie Valley", "Jeff"
**previous tokens negatively activating tokens**: "to", "North", "to", "as", "North", "the", "."

Step 2.
The positively activating tokens reference the East Coast of the United States. One explanation for the - in front of the "050" suggests the neuron might be recognizing the relevance to GMT -0500, which is the East Coast time zone.
The negatively activating tokens reference locations on the West Coast of North America. Snoqualmie Valley is a location in Western Washington. British Columbia is a Western Canadian province, suggesting the neuron is not specific to locations in the United States. " western" and " America" (preceded by North) suggest contexts in the Western part of North America. Idaho is also consistent with this pattern. Jeff could refer to Jeff Bezos, who is from Washington.

Step 3.
- Positively activating examples are on the East Coast of North America
- Negatively activating examples are on the West coast of North America.

Let me think carefully, did I miss anything?
Yes, negatively activating examples seem to specifically be in the Pacific Northwest region.

Step 4.
[POSITIVE DESCRIPTION]: Locations on the East Coast of North America,
[NEGATIVE DESCRIPTION]: Locations on the West Coast of North America, specifically in the Pacific Northwest.""",
        ),
    ]
]

SYS_PROMPT_DEFAULT = """\
You are a meticulous AI researcher conducting an important investigation into a specific neuron inside a language model that activates in response to text excerpts.

Your overall task is to describe features of text excerpts that cause the neuron to strongly activate. To do this, you will be provided with text excerpts that maximally activate the neuron. Use the following instructions to process this information to produce an accurate and concise description of the neuron's behavior.

## Instructions

You will receive a list of text excerpts on which the neuron activates. Tokens causing activation will appear between delimiters like {{this}}. Consecutive activating tokens will also be accordingly delimited {{just like this}}. Use the following steps to produce a description of the text features that cause activations.

## Steps to produce a description of neuron behavior

1. **Analyze Each Excerpt Individually:**
   - For each text excerpt provided, note which tokens (words, fragments of words, or symbols) caused the neuron to activate.
   - Note the context, preceding, and following tokens closely associated with each activating token.

2. **Identify Patterns:**
   - Examine the noted tokens from Step 1 to find recurring patterns or commonalities.

3. **Summarize Shared Features:**
   - List general features shared among the text excerpts that produce activations. Focus on key similarities pertinent to activation.

3. **Formulate Descriptions:**
   - Utilize the identified patterns and shared features to construct a description about the features of the text that lead to neuron activation.
   - Aim to produce as succinct of descriptions as possible; the descriptions should distinctly highlight what causes activation.
   - Produce your final descriptions in the following format; it's very important that you stick to this exact format:
     [DESCRIPTION]: <final description of the activations>

## Guidelines:

- Try to be as specific as possible without listing examples of words.
- Avoid using words like "often", "particularly" or "especially" in your explanations. Either a feature is relevant to the explanation, or it isn't. There is no in between.
- Think very carefully and show your work.
- The list of excerpts will provide a lot of information. Some of it might not be very useful. Try to focus on overall patterns that you see, and do not overfit to try to accommodate all the data. Focus on the most obvious patterns.
- Try to produce concise final descriptions. Simply describe the text features that activate the neuron."""

EXAMPLES_DEFAULT = [
    [
        ChatMessage(
            role="user",
            content="""\
Excerpt 1:  This species of bumble bee is native to {{western}} North {{America}} and is found from California to {{British Columbia}} and as far east as {{Idaho}}.
Excerpt 2:  North {{Bend}} is paying forward the support the {{Snoqualmie Valley}} has shown them.
{{Jeff}} and Teresa Warren came here about a year ago,
Excerpt 3:  I know Hindi, Marathi & Gujarati Languages and I can mail you the translated file by this weekend. You can mail me on this ID. Date: Sat, 3 Jan 2009 19:52:55 -0500""",
        ),
        ChatMessage(
            role="assistant",
            content="""\
Step 1.
**activating tokens:** "western", "America", "British Columbia", "Idaho", "Snoqualmie Valley", "Jeff"
**previous tokens to activating tokens**: "to", "North", "to", "as", "North", "the", "."
**next tokens to activating tokens**: "North", "and", "and", ".", "is", "has", "and"

Step 2.
The activating tokens reference locations on the West Coast of North America. Snoqualmie Valley is a location in Western Washington. British Columbia is a Western Canadian province, suggesting the neuron is not specific to locations in the United States. " western" and " America" (preceded by North) suggest contexts in the Western part of North America. Idaho is also consistent with this pattern. Jeff could refer to Jeff Bezos, who is from Washington.

Step 3.
- activating examples are on the West coast of North America.

Let me think carefully, did I miss anything?
Yes, activating examples seem to specifically be in the Pacific Northwest region.

Step 4.
[DESCRIPTION]: Locations on the West Coast of North America, specifically in the Pacific Northwest.""",
        ),
    ],
    [
        ChatMessage(
            role="user",
            content="""\
Excerpt 1:  I climbed a pear tree and picked a pear. I climbed an apple tree and {{picked}} an apple.
Excerpt 2: Looking for an easy way to protest Bush foreign policy week after week? {{And}} an easy way to help alleviate global poverty? Buy your gasoline at Citgo stations. Looking {{for}} an easy way to protest Bush foreign policy week after week? {{And}} an easy way to help alleviate global poverty?
Excerpt 3: At one point, the trooper said the car was going over 100 {{miles}} an hour.""",
        ),
        ChatMessage(
            role="assistant",
            content="""\
Step 1.
**activating tokens:** "picked", "And", "for", "And", "miles"
**previous tokens to activating tokens**: "and", "week?", "Looking", "week?", "100"
**next tokens to activating tokens**: "an", "an", "an", "an", "an"

Step 2.
- The tokens "picked", "for", and "miles" are action-oriented words.
- The token "And" appears to act as a conjunction connecting phrases, but is repeated in the context. The repeated structure may be relevant.
- These tokens appeared in contexts where they were followed by "an" which might be a significant clue.

Step 3.
- All activating tokens have "an" as the subsequent token.

Step 4.
[DESCRIPTION]: positions in the sentence where the next word is likely to be "an\"""",
        ),
    ],
]

SYS_PROMPT_UPDATED = """\
You are a meticulous AI researcher conducting an important investigation into a specific neuron inside a language model that activates in response to text excerpts.

Your overall task is to describe features of text excerpts that cause the neuron to strongly activate. To do this, you will be provided with text excerpts that maximally activate the neuron. Use the following instructions to process this information to produce an accurate and concise description of the neuron's behavior.

## Instructions

You will receive a list of text excerpts on which the neuron activates. Tokens causing activation will appear between delimiters like {{this}}. Consecutive activating tokens will also be accordingly delimited {{just like this}}. It's important to note that language models read a text left to right, meaning that a neuron activates on a token based on information previous to that token as opposed to the whole text.

## Steps to produce a description of neuron behavior

1. **Analyze Each Excerpt Individually:**
    - For each text excerpt provided, note which tokens (words, fragments of words, or symbols) caused the neuron to activate.
    - Note the context, preceding, and following tokens closely associated with each activating token. It is important to look both at the tokens immediately before and after the activating token, as well as the overall context in which the tokens occur.

2. **Identify Patterns in Max Activating Tokens:**
    - Examine the noted tokens from Step 1 to find recurring patterns or commonalities.

3. **Summarize Commonalities in Excerpts:**
    - Describe features shared among the excerpts in which max activations occur. Focus on key similarities pertinent to activation.
    - The shared feature might be related to the content (e.g. excerpts are all related to a certain topic), but it might also be related to the structure (e.g. excerpts all have repetitive parts.)

4. **Identify Patterns within Entire Context**
    - Examine the entire excerpt in which the max activating token occurs. There may be patterns that exist outside of the tokens preceding and following the max activating token. These patterns may reflect previous times earlier in the excerpt that the token appeared, or other content earlier in the excerpt that is relevant to the max activating token.

5. **Incorporate all Findings:**
    - Utilize all identified patterns and shared features from steps 1-4 to construct a description about the features of the text that lead to neuron activation.
    - Remember to specifically describe the patterns in the max activating tokens surrounded by delimiters like {{this}}. You may use patterns in the surrounding tokens (the preceding and following tokens, as well as the broader excerpt) to characterize the max activating tokens, but be sure to distinguish the max activating tokens from surrounding context in your description (in other words, do not confuse the max activating tokens and tokens nearby).
    - Aim to produce as succinct of descriptions as possible; the descriptions should distinctly highlight what causes activation.

6. **Present Final Description:**
    - Produce your final description in the following format; it's very important that you stick to this exact format:
    [DESCRIPTION]: <final description of the activations>

## Overall Guidelines:

- Try to be as specific as possible without listing examples of words.
- Avoid using words like "often", "particularly" or "especially" in your explanations. Either a feature is relevant to the explanation, or it isn't. There is no in between.
- Think very carefully and show your work.
- The list of excerpts will provide a lot of information. Some of it might not be very useful. Try to focus on overall patterns that you see, and do not overfit to try to accommodate all the data. Focus on the most obvious patterns.
- Try to produce concise final descriptions. Simply describe the text features that activate the neuron."""

EXAMPLES_UPDATED = [
    [
        ChatMessage(
            role="user",
            content="""\
Excerpt 1:  This species of bumble bee is native to {{western}} North {{America}} and is found from California to {{British Columbia}} and as far east as {{Idaho}}.
Excerpt 2:  North Bend is paying forward the support the {{Snoqualmie Valley}} has shown them.
{{Jeff}} and Teresa Warren came here about a year ago,
Excerpt 3:  The Sequim Gazette is located at 147 W. Washington Street in Sequim. Business hours are Monday through Friday from 8:30 a.m. to 5:00 p.m. Phone {{360}}-683-3311, or toll free at 800-829-5810.""",
        ),
        ChatMessage(
            role="assistant",
            content="""\
Step 1.
**activating tokens:** "western", "America", "British Columbia", "Idaho", "Snoqualmie Valley", "Jeff", "360"
**previous tokens to activating tokens**: "to", "North", "to", "as", "the", ".", "Phone"
**next tokens to activating tokens**: "North", "and", "and", ".", "has", "and", "-683-3311"

Step 2.
- The activating tokens reference locations on the Pacific Northwest region. Snoqualmie Valley is a location in Western Washington. British Columbia is a Western Canadian province, suggesting the neuron is not specific to locations in the United States. "western" and "America" (preceded by North) suggest contexts in the Western part of North America. Idaho is also consistent with this pattern. Jeff could refer to Jeff Bezos, who is from Washington. 360 is the area code for used in Washington.

Step 3.
- All excerpts reference the Pacific Northwest region, either directly or indirectly.

Step 4.
- There are no obvious patterns regarding the entire context of the excerpts.

Step 5.
- There is strong evidence for the neuron activating at tokens related to the Pacific Northwest region.

Step 6.
[DESCRIPTION]: References to the Pacific Northwest region.""",
        ),
    ],
    [
        ChatMessage(
            role="user",
            content="""\
Excerpt 1:  I climbed a pear tree and picked a pear. I climbed an apple tree and {{picked}} an apple.
Excerpt 2: Looking for an easy way to protest Bush foreign policy week after week? {{And}} an easy way to help alleviate global poverty? Buy your gasoline at Citgo stations. Looking {{for}} an easy way to protest Bush foreign policy week after week? {{And}} an easy way to help alleviate global poverty?
Excerpt 3: At one point, the trooper said the car was going over 100 {{miles}} an hour.
Excerpt 4: It took me a long time to fall asleep last night. I laid in bed {{for}} {{almost}} an hour.""",
        ),
        ChatMessage(
            role="assistant",
            content="""\
Step 1.
**activating tokens:** "picked", "And", "for", "And", "miles", "for", "almost"
**previous tokens to activating tokens**: "and", "week?", "Looking", "week?", "100", "bed", "for"
**next tokens to activating tokens**: "an", "an", "an", "an", "an", "almost", "an",

Step 2.
- All activating tokens were followed by "an" (with the exception of "almost") which might be a significant clue.

Step 3.
- The excerpts don't seem to have much in common regarding the content.
- The first two excerpts contain sentences that are repeated.

Step 4.
- All activating tokens are in contexts such that it is likely to be followed by an "an". In excerpt 1 and excerpt 2, the repeating sentence makes it likely that "picked" and "for" are followed by an "an", while in the rest of the excerpts "an" is natural to come after "miles", "for" and "almost".

Step 5.
Given all the previous evidence, it's likely that the neuron activates on positions in the sentence where the next word is likely to be an "an".

Step 6.
[DESCRIPTION]: positions in the sentence where the next word is likely to be "an\"""",
        ),
    ],
    [
        ChatMessage(
            role="user",
            content="""\
Excerpt 1:  Once upon a time, there was a brave knight named Art who lived in a magnificent castle. One day, a beautiful princess named Gwen came to visit the kingdom. {{Art}} was immediately captivated by her beauty and grace.
Excerpt 2:  Another time, when Silas left the city together with Giles, to fetch some livestock for their brothers and teachers, {{Silas}} began to speak.
Excerpt 3:  Once upon a time, there was a brave knight named Arthur who lived in a magnificent castle. One day, {{Arthur}} went to visit another kingdom. While he was there, he went on many adventures and faced many perils, but he always came through. Arthur returned to his castle and lived happily ever after.
Excerpt 4:  Ben enjoyed apples, while Alice enjoyed bananas. {{Alice}} found it funny that the first letter of their favorite fruits matched the first letter of the other's name, but {{Ben}} did not.
Excerpt 5:  Kate, Eve, and Claire were all siblings. {{Kate}} was the oldest, {{Eve}} the second oldest, and {{Claire}} the youngest.
Excerpt 6:  Chocolate cake or apple pie - Sam couldn't decide which he liked better.
Excerpt 7:  Tess, Jack, Mark, and Jane were bored on a rainy Saturday afternoon. They had played all their board games, watched all their favorite movies, and read all their books. They wanted to do something fun and exciting, but they couldn't go outside or visit their friends. {{Tess}} decided to make up an adventure game, using her imagination and whatever she could find in the house. {{Jack}} turned the living room into a jungle, the kitchen into a spaceship, the basement into a dungeon, and the attic into a treasure island. {{Mark}} pretended to be an explorer, then an astronaut.""",
        ),
        ChatMessage(
            role="assistant",
            content="""\
Step 1.
**activating tokens:** "Art", "Silas", "Arthur", "Alice", "Ben", "Kate", "Eve", "Claire", "Tess", "Jack", "Mark"
**previous tokens to activating tokens**: ".", ",", ",", ".", "but", ".", ",", "and", ".", ".", "."
**next tokens to activating tokens**: "was ", "began", "went", "found", "did", "was", "the", "the", "decided", "turned", "pretended"

Step 2.
Activations occur on certain names. I do not see a pattern in the names themselves, or in the previous or next tokens.

Step 3.
The max activations all occur inside excerpts that resemble stories with one or more characters.

Step 4.
Looking earlier in the excerpts, I see additional occurrences of the names. It appears that activations occur the second time a name appears.

Step 5.
The main pattern found was from step 4 that activations occur when a character is mentioned in a story the second time.

Step 6.
[DESCRIPTION]: The second time a character is mentioned in a story.""",
        ),
    ],
    [
        ChatMessage(
            role="user",
            content="""\
Excerpt 1:  There was snow everywhere. It made me so happy that I jumped {{up}} and down.
Excerpt 2:  The doctors told me it was too late-- I would {{die}} in less than a month.
Excerpt 3:  "Should I bake the cake now?" I thought {{to}} myself.
Excerpt 4:  The dance would go like this: he would jump up and I would {{jump}} down.
Excerpt 5:  I couldn't {{bear}} to see him cry. It was too painful.""",
        ),
        ChatMessage(
            role="assistant",
            content="""\
Step 1.
**activating tokens:** "up", "die", "to", "jump", "bear"
**previous tokens to activating tokens**: "jumped", "would", "thought", "would", "couldn't"
**next tokens to activating tokens**: "and", "in", "myself", "down", "to"

Step 2.
- All previous tokens are verbs or auxiliary verbs.

Step 3.
- The excerpts are all in a first-person narrative.

Step 4.
- Neuron activation appears tied to phrases involving "I", specifically when the token is positioned after "I [verb]".

Step 5.
From step 4, it seems that the neuron activates on the word immediately after "I" followed by a verb.

Step 6.
[DESCRIPTION]: Only the word that comes after "I [verb]".""",
        ),
    ],
]


SYS_PROMPT_NO_COT = """\
You are a meticulous AI researcher conducting an important investigation into a specific neuron inside a language model that activates in response to text excerpts. Your overall task is to describe features of text excerpts that cause the neuron to strongly activate.

You will receive a list of text excerpts on which the neuron activates. Tokens causing activation will appear between delimiters like {{this}}. Consecutive activating tokens will also be accordingly delimited {{just like this}}. If no tokens are highlighted with {{}}, then the neuron does not activate on any tokens in the excerpt.

Note: Neurons activate on a word-by-word basis. Also, neuron activations can only depend on words before the word it activates on, so the description cannot depend on words that come after, and should only depend on words that come before the activation."""


SYS_PROMPT_FOR_ACTIVATION_VALUES = """\
You are a meticulous AI researcher conducting an important investigation into a specific neuron inside a language model that activates in response to text excerpts.

Your overall task is to describe features of input text that cause the neuron to strongly activate. To do this, you will be provided with text excerpts that maximally activate the neuron, and activation values on those inputs.

Note: Neurons activate on a word-by-word basis. Also, neuron activations can only depend on words before the word it activates on, so the description cannot depend on words that come after, and should only depend on words that come before the activation."""


SYS_PROMPT_BILLS = """\
We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is looking for. Don't list examples of words.

The activation format is token<tab>activation. Activation values range from 0 to 10. A neuron finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match."""
