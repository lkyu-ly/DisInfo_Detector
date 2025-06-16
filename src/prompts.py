DECOMPOSE_NUM_PROMPT_TEMPLATE = """
Fact-checking involves accessing the veracity of a given text, which could be a statement, claim, paragraph, or a generated response from a large language model.

For more complex fact-checking tasks, the input text is often broken down into a set of manageable, verifiable and self-contained claims, a process called 'Input Decomposition'. Each claim is required to be *self-contained*, meaning it is completely interpretable without the original text and other claims. Each claim is verified independently, and the results are combined to assess the overall veracity of the original input.

Your task is to decompose an input text into exactly {num_sub_claims} claims while preserving the original meaning. Each claim should be *self-contained*, meaning it is completely interpretable without the original text and other claims.

Please directly provide the decomposed set of claims in the following format, each claim should be enclosed with triple backticks:

### Claims
```
claim #1
```

```
claim #2
```
...
```
claim #{num_sub_claims}
```

Now, please help me decompose the following input text:

### Input Text
```
{input_text}
```
"""


DECOMPOSE_SINGLE_PROMPT_TEMPLATE = """
Please summarize the following input text into a single sentence of 20 tokens or fewer. The summary should capture the essence of the original text.

### Input Text
```
{input_text}
```

Please provide the summary directly without any additional openings, closings, or explanations.
"""


EXTRACTION_NON_QA_TEMPLATE = """You are trying to verify how factual a piece of text is. To do so, you need to break down a sentence and extract as many fine-grained facts mentioned in the sentence as possible. Each of these fine-grained facts should be verifiable against reliable external world knowledge (e.g., via Wikipedia). Any story, personal experiences, hypotheticals (e.g., "would be" or subjunctive), subjective statements (e.g., opinions), suggestions, advice, instructions, and other such content should not be included in the list. Biographical, historical, scientific, and other such texts are not personal experiences or stories. You should extract verifiable facts from them. Each fact should also be describing either one single event (e.g., "Nvidia is founded in 1993 in Sunnyvale, California, U.S.") or single state (e.g., "UMass Amherst has existed for 161 years.") with necessary time and location information. Quotations should be extracted verbatim with the source when available. Listed references should be ignored.

Extract fine-grained facts from the sentence marked between <SOS> and <EOS>. You should focus on the named entities and numbers in the sentence and extract relevant information from the sentence. Other sentences are only context for you to recover pronouns, definite phrases (e.g., "the victims" or "the pope"), and so on. Each fact should be understandable on its own and require no additional context. This means that all entities must be referred to by name but not pronoun. Use the name of entities rather than definite noun phrases (e.g., 'the teacher') whenever possible. If a definite noun phrase is used, be sure to add modifiers (e.g., a embedded clause, a prepositional phrase, etc.). Each fact must be situated within relevant temporal and location whenever needed. Keep each fact to one sentence with zero or at most one embedded clause. You do not need to justify what you extract.

If there is no verifiable fact in the sentence, please write "No verifiable claim."

Here are some examples:

Text: The sweet potato or sweetpotato (Ipomoea batatas) is a dicotyledonous plant that belongs to the bindweed or morning glory family, Convolvulaceae. <SOS>Its large, starchy, sweet-tasting tuberous roots are used as a root vegetable.<EOS> The young shoots and leaves are sometimes eaten as greens.
Sentence to be focused on: Its large, starchy, sweet-tasting tuberous roots are used as a root vegetable.
Facts:
- Sweet potatoes' roots are large.
- Sweet potatoes' roots are starchy.
- Sweet potatoes' roots are sweet-tasting.
- Sweet potatoes' roots are tuberous.
- Sweet potatoes' roots are used as a root vegetable.

Text: <SOS>After the success of the David in 1504, Michelangelo’s work consisted almost entirely of vast projects.<EOS> He was attracted to these ambitious tasks while at the same time rejecting the use of assistants, so that most of these projects were impractical and remained unfinished.
Sentence to be focused on: After the success of the David in 1504, Michelangelo’s work consisted almost entirely of vast projects.
Facts:
- Michelangelo achieved the success of the David in 1504.
- After 1504, Michelangelo’s work consisted almost entirely of vast projects.

Text: After the success of the David in 1504, Michelangelo’s work consisted almost entirely of vast projects. He was attracted to these ambitious tasks while at the same time rejecting the use of assistants, so that most of these projects were impractical and remained unfinished. <SOS>In 1504 he agreed to paint a huge fresco for the Sala del Gran Consiglio of the Florence city hall to form a pair with another just begun by Leonardo da Vinci.<EOS> Both murals recorded military victories by the city (Michelangelo’s was the Battle of Cascina), but each also gave testimony to the special skills of the city’s much vaunted artists.
Sentence to be focused on: In 1504 he agreed to paint a huge fresco for the Sala del Gran Consiglio of the Florence city hall to form a pair with another just begun by Leonardo da Vinci.
Facts:
- In 1504, Michelangelo agreed to paint a huge fresco for the Sala del Gran Consiglio of the Florence city hall.
- Around 1504, Leonardo da Vinci just began with a mural for the Florence city hall.

Text: After the success of the David in 1504, Michelangelo’s work consisted almost entirely of vast projects. He was attracted to these ambitious tasks while at the same time rejecting the use of assistants, so that most of these projects were impractical and remained unfinished. In 1504 he agreed to paint a huge fresco for the Sala del Gran Consiglio of the Florence city hall to form a pair with another just begun by Leonardo da Vinci. <SOS>Both murals recorded military victories by the city (Michelangelo’s was the Battle of Cascina), but each also gave testimony to the special skills of the city’s much vaunted artists.<EOS> Leonardo’s design shows galloping horses, Michelangelo’s active nudes—soldiers stop swimming and climb out of a river to answer an alarm.
Sentence to be focused on: Both murals recorded military victories by the city (Michelangelo’s was the Battle of Cascina), but each also gave testimony to the special skills of the city’s much vaunted artists.
Facts:
- Michelangelo’s murals for the Florence city hall recorded military victories by the city.
- Leonardo da Vinci’s murals for the Florence city hall recorded military victories by the city.
- Michelangelo’s mural for the Florence city hall was the Battle of Cascina.

Text: I (27f) and my fiance "Leo" (27m) decided to let my FSIL "Maya" (32f) stay at our house because she needed space from her husband due to some relationship struggles they're having. Leo and I had gotten wedding cake samples from an expensive bakery specializing in wedding cakes. We planned to test them along with Maya after we finished up some other wedding plans yesterday. <SOS>However, when I came home from work to see Leo yelling at Maya, the box the samples came in wide open on the living room table, and Maya arguing with him.<EOS> I asked what was happening, and Leo angrily told me that while we were both at work, Maya had some friends over and they ended up eating almost all of our cake samples.
Sentence to be focused on: However, when I came home from work to see Leo yelling at Maya, the box the samples came in wide open on the living room table, and Maya arguing with him.
Facts:
No verifiable claim.

Text: I was a catholic school kid, educated by nuns and somehow on a spring day in 1972, I was called down to the principal’s office by Sister Mary Roberts, who informed me that I had gained admission to Stuyvesant High School. <SOS>I was excited to be freshman in one of New York City’s elite public schools but soon came to realize that my catholic school education did not provide the groundwork for abstract concepts like science and algebra.<EOS> My parochial education in Science at St. Joseph’s was essentially “God made it, what else do you need to know?”
Sentence to be focused on: I was excited to be freshman in one of New York City’s elite public schools but soon came to realize that my catholic school education did not provide the groundwork for abstract concepts like science and algebra.
Facts:
- Stuyvesant High School is in New York City.
- Stuyvesant High School is an elite high school.
- Stuyvesant High School is a public school.
- In 1972, St. Joseph's catholic school education did not provide the groundwork for abstract concepts like science and algebra.

Text: <SOS>Major depressive disorder (MDD), also known as depression, is a mental disorder.<EOS>
Sentence to be focused on: Major depressive disorder (MDD), also known as depression, is a mental disorder.
Facts:
- Major depressive disorder is also known as depression.
- Major depressive disorder is a mental disorder.

Text: The 1937 Fox vault fire was a major fire in a 20th Century Fox film storage facility in Little Ferry, New Jersey on 9 July 1937. It was caused by the spontaneous combustion of nitrate film stored in inadequately-ventilated vaults. The fire resulted in one death and two injuries, and destroyed all of the film present. <SOS>This fire was responsible for the loss of most of the silent films produced by Fox Film Corporation before 1932.<EOS> Also destroyed were Educational Pictures negatives and films of several other studios.
Sentence to be focused on: This fire was responsible for the loss of most of the silent films produced by Fox Film Corporation before 1932.
Facts:
- Fox Film Corporation produced silent films before 1932.
- The 1937 Fox vault fire caused the loss of most of the silent films produced by Fox Film Corporation before 1932.

Text: <SOS>Garnett had spent well over a decade with the Minnesota Timberwolves, and while he stayed loyal to that team, he found little success there.<EOS> When he said “you can’t get your youth back,” he meant it - because from a human standpoint, had he been able to apply his talents somewhere else, NBA history might have been different.
Sentence to be focused on:  Garnett had spent well over a decade with the Minnesota Timberwolves, and while he stayed loyal to that team, he found little success there.
Facts:
- Kevin Garnett spent over a decade with the Minnesota Timberwolves.
- Kevin Garnett was loyal to the Minnesota Timberwolves.
- Kevin Garnett found little success with the Minnesota Timberwolves.

Text: Garnett had spent well over a decade with the Minnesota Timberwolves, and while he stayed loyal to that team, he found little success there. <SOS>When he said “you can’t get your youth back,” he meant it - because from a human standpoint, had he been able to apply his talents somewhere else, NBA history might have been different.<EOS>
Sentence to be focused on: When he said “you can’t get your youth back,” he meant it - because from a human standpoint, had he been able to apply his talents somewhere else, NBA history might have been different.
Facts:
- Kevin Garnett said "you can’t get your youth back."

Text: Unity. Unity. In another January in Washington, on New Year’s Day 1863, Abraham Lincoln signed the Emancipation Proclamation. <SOS>When he put pen to paper, the President said, “If my name ever goes down into history it will be for this act and my whole soul is in it.”<EOS> My whole soul is in it.
Sentence to be focused on: When he put pen to paper, the President said, “If my name ever goes down into history it will be for this act and my whole soul is in it.”
Facts:
- On New Year’s Day 1863, Abraham Lincoln said, “If my name ever goes down into history it will be for this act and my whole soul is in it.”

Text: Ãcariya Mun related the story of a dhutanga monk (ascetic monk) who inadvertently went to stay in a forest located next to a charnel ground. He arrived on foot at a certain village late one afternoon and, being unfamiliar with the area, asked the villagers where he could find a wooded area suitable for meditation. They pointed to a tract of forest, claiming it was suitable, but neglected to tell him that it was situated right on the edge of a charnel ground. <SOS>They then guided him to the forest, where he passed the first night peacefully.<EOS> On the following day he saw the villagers pass by carrying a corpse, which they soon cremated only a short distance from where he was staying.
Sentence to be focused on: They then guided him to the forest, where he passed the first night peacefully.
Facts:
No verifiable claim.

Text: <SOS>The sweet potato or sweetpotato (Ipomoea batatas) is a dicotyledonous plant that belongs to the bindweed or morning glory family, Convolvulaceae.<EOS> Its large, starchy, sweet-tasting tuberous roots are used as a root vegetable. The young shoots and leaves are sometimes eaten as greens.
Sentence to be focused on: The sweet potato or sweetpotato (Ipomoea batatas) is a dicotyledonous plant that belongs to the bindweed or morning glory family, Convolvulaceae.
Facts:
- The scientific name of sweet potatoes is Ipomoea batatas.
- Sweet potatoes are dicotyledonous plants.
- Sweet potatoes belong to the bindweed or morning glory family, Convolvulaceae.

Text: Pope Julius had an ambitious imagination, parallel to Michelangelo’s, but because of other projects, such as the new building of St. Peter’s and his military campaigns, he evidently became disturbed soon by the cost. Michelangelo believed that Bramante, the equally prestigious architect at St. Peter’s, had influenced the pope to cut off his funds. He left Rome, but the pope brought pressure on the city authorities of Florence to send him back. <SOS>He was put to work on a colossal bronze statue of the pope in his newly conquered city of Bologna (which the citizens pulled down soon after when they drove the papal army out) and then on the less expensive project of painting the ceiling of the Sistine Chapel (1508–12).<EOS>
Sentence to be focused on: He was put to work on a colossal bronze statue of the pope in his newly conquered city of Bologna (which the citizens pulled down soon after when they drove the papal army out) and then on the less expensive project of painting the ceiling of the Sistine Chapel (1508–12).
Facts:
- Michelangelo was put to work on a colossal bronze statue of Pope Julius II.
- The city of Bologna was once conquered by Pope Julius II.
- The colossal bronze statue of Pope Julius II was put in Bologna.
- The papal army was driven out of Bologna.
- The citizens of the Bologna pulled down the bronze statue of Pope Julius II after they drove the papal army out.
- Michelangelo worked on painting the ceiling of the Sistine Chapel from 1508 to 1512.

Extract *verifiable atomic* facts.

Text: {snippet}
Sentence to be focused on: {sentence}
Facts:"""

WICE_PROMPT = f"""
Segment the following sentence into individual facts:

Sentence: Other title changes included Lord Steven Regal and The Nasty Boys winning the World Television Championship and the World Tag Team Championship respectively.
Facts:
- Lord Steven Regal wan the World Television Championship. 
- The Nasty Boys wan and the World Tag Team Championship.

Sentence: The parkway was opened in 2001 after just under a year of construction and almost two decades of community requests.
Facts:
- The parkway was opened in 2001.
- The parkway was opened after just under a year of construction.
- The parkway was opened after two decades of community requests.

Sentence: Touring began in Europe in April–June with guitarist Paul Gilbert as the opening act, followed by Australia and New Zealand in July, Mexico and South America in late July–August, and concluding in North America in October–November.
Facts:
- Touring began in Europe in April–June.
- The opening act was guitarist Paul Gilbert.
- There was a tour in Australia in July.
- There was a tour in New Zealand in July.
- There was a tour in Mexico in late July–August.
- There was a tour in South America in late July–August
- The tour was concluded in North America in October–November.

Sentence: In March 2018, the company partnered With Amazon Web Services (AWS) to offer Al-enabled conversational solutions to customers in India.
Facts:
- The company partnered with Amazon Web Services (AWS) in March 2018.
- The two companies partnered to offer Al-enabled conversational solutions to customers in India.

Sentence: The most significant of these is in Germany, which now has a Yazidi community of more than 200,000 living primarily in Hannover, Bielefeld, Celle, Bremen, Bad Oeynhausen, Pforzheim and Oldenburg.
Facts:
- The most significant of these is in Germany.
- Germany now has a Yazidi community of more than 200,000.
- Yazidi community in Germany lives primarily in Hannover.
- Yazidi community in Germany lives primarily in Bielefeld.
- Yazidi community in Germany lives primarily in Celle.
- Yazidi community in Germany lives primarily in Bremen.
- Yazidi community in Germany lives primarily in Bad Oeynhausen.
- Yazidi community in Germany lives primarily in Pforzheim.
- Yazidi community in Germany lives primarily in Oldenburg.

Sentence: A previous six-time winner of the Nations' Cup, Sebastian Vettel became Champion of Champions for the first time, defeating Tom Kristensen, who made the final for the fourth time, 2–0.
Facts:
- Sebastian Vettel is a previous six-time winner of the Nations' Cup.
- Sebastian Vettel became Champion of Champions for the first time.
- Sebastian Vettel defeated Tom Kristensen.
- Tom Kristensen made the final for the fourth time.
- The score was 2–0.

Sentence: {{}}
Facts:\n"""

VERIFICATION_INSTRUCTION_BINARY_NLI = """You need to judge whether a claim is supported or not by Google search results. When doing the task, take into consideration whether the link of the search result is of a trustworthy source. 

Below are the definitions of the two categories:

Supported: A claim is supported by the search results if everything in the claim is supported and nothing is contradicted by the search results. There can be some search results that are not fully related to the claim.
Unsupported: If a claim is not supported by the search results, mark it as unsupported.

For your decision, you must only output the word 'Supported', or the word 'Unsupported', nothing else.

Here are some examples:

Claim: {}

{}

Your decision: {}

Claim: {}

{}

Your decision: {}

Claim: {}

{}

Your decision: {}

Claim: {}

{}

Your decision: {}

Claim: {}

{}

Your decision: {}

Claim: {}

{}

Your decision: {}

Claim: {}

{}

Your decision: {}

Claim: {}

{}

Your decision: {}

Claim: {}

{}

Your decision: {}"""

VERIFICATION_INSTRUCTION_TRINARY = """You need to judge whether a claim is supported or contradicted by Google search results, or whether there is no enough information to make the judgement. When doing the task, take into consideration whether the link of the search result is of a trustworthy source. Mark your answer with ### signs.

Below are the definitions of the three categories:

Supported: A claim is supported by the search results if everything in the claim is supported and nothing is contradicted by the search results. There can be some search results that are not fully related to the claim.
Contradicted: A claim is contradicted by the search results if something in the claim is contradicted by some search results. There should be no search result that supports the same part.
Inconclusive: A claim is inconclusive based on the search results if:
- a part of a claim cannot be verified by the search results,
- a part of a claim is supported and contradicted by different pieces of evidence,
- the entity/person mentioned in the claim has no clear referent (e.g., "the approach", "Emily", "a book").

Here are some examples:

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###

Claim: {}

{}

Your decision: ###{}###"""