# LLM comparison — answers

Generated 2026-08-09T23:34:03. Session `eval_set`, `n_results=5`.

Retrieval ran **once per question**; the identical retrieved chunks and the identical prompt were sent to all three models, so the model is the only variable. The prompt is the production one (`backend.retrieve.build_rag_prompt`), which instructs the model to answer using only the supplied notes.

Answers are recorded verbatim and are **not** judged here.

Models: `llama3.2`, `mistral:7b`, `llama3:8b`

---

## Q4 — If a word's dictionary root differs from chopping its suffix, which technique finds the root?

<details><summary>Retrieved chunks (shared by all models)</summary>

**1. `Natural_Language_Processing.mp3` #5** — distance 0.4899

```text
we've got that, the first stage of NLP is called tokenization. This is about taking a string and breaking it down into chunks. So, if we consider the unstructured text we've got here, add eggs and milk to my shopping list. That's eight words. That could be eight tokens. And from here on in, we are going to work one token at a time as we traverse through this. Now, the first stage once we've got things down into tokens that we can perform is called stemming. And this is all about deriving the word stem for a given token. So, for example, running runs and ran. The word stem for all three of those is run. We're just kind of removing the prefix and the suffixes and normalizing the tense and we're getting to the word stem. But stemming doesn't work well for every token. For example, universal and university will, well, they don't really stem down to universe. For situations like that, there is another tool that we have available and that is called Lematization. And lematization takes a given token and learns its meaning through a dictionary. Definition and from there it can derive its root or
```

**2. `Natural_Language_Processing.mp3` #6** — distance 0.5395

```text
university will, well, they don't really stem down to universe. For situations like that, there is another tool that we have available and that is called Lematization. And lematization takes a given token and learns its meaning through a dictionary. Definition and from there it can derive its root or its Lem. So, take better, for example, better is derived from good. So, the root or the Lem of better is good. The stem of better would be bet. So, you can see that it is significant whether we use stemming or we use Lematization for a given token. Now, next thing we can do is we can do a process called part of speech tagging. And what this is doing is for a given token is looking where that token is used within the context of a sentence. So, take the word make, for example. If I say, I'm going to make dinner, make is a verb. But if I ask you what make is your laptop, well, make is now now. So, where that token is used in the sentence matters. Part of speech tagging can help us derive that context. And then finally, another stage is named
```

**3. `Natural_Language_Processing.mp3` #4** — distance 0.7578

```text
example is spam detection. So, this is a case of looking at a given email message and trying to derive, is this a really email message or is it spam? And we can look for pointers within the content of the message. So, things like overused words or poor grammar or an inappropriate claim of urgency can all indicate that this is actually, perhaps, spam. So, those are some of the things that NLP can provide, but how does it work? Well, the thing with NLP is it's not like one algorithm. It's actually more like a bag of tools. And you can apply these bag of tools to be able to resolve some of these use cases. Now, the input to NLP is some unstructured text, so either some written text or spoken text that has been converted to a written text through a speech to text algorithm. Once we've got that, the first stage of NLP is called tokenization. This is about taking a string and breaking it down into chunks. So, if we consider the unstructured text we've got here, add eggs and milk to my shopping list. That's eight words. That could be eight tokens. And
```

**4. `Natural_Language_Processing.mp3` #1** — distance 0.7888

```text
a structured representation of that same information that a computer can process. Now, that might look something a bit more like this, where we have a shopping list element. And then it has sub elements within it, like an item for eggs. And an item for milk. That is an example of something that is structured. Now, the job of natural language processing is to translate between these two things. So, NLP sits right in the middle here, translating between unstructured and structured data. And when we go from unstructured here to structured this way, that's called NLU or natural language understanding. And when we go this way from structured to unstructured, that's called natural language generation or NLG. We're going to focus today primarily on going from unstructured to structured in natural language processing. Now, let's think of some use cases where NLP might be quite handy. First of all, we've got machine translation. Now, when we translate from one language to another, we need to understand the context of that sentence. It's not just the case of taking each individual word from, say, English and then translating it into another language. We need to understand the overall structure
```

**5. `Natural_Language_Processing.mp3` #2** — distance 0.7951

```text
all, we've got machine translation. Now, when we translate from one language to another, we need to understand the context of that sentence. It's not just the case of taking each individual word from, say, English and then translating it into another language. We need to understand the overall structure and context of what's being said. And my favorite example of this going horribly wrong is if you take the phrase, the spirit is willing, but the flesh is weak. And you translate that from English to Russian, and then you translate that Russian translation back into English, you're going to go from the spirit is willing, but the flesh is weak to something a bit more like the vodka is good, but the meat is rotten, which is really not the intended context of that sentence, whatsoever. So, NLP can help with situations like that. Now, the second kind of use case that I like to mention relates to virtual assistance. And also to things like chatbots. Now, a virtual assistant, that's something like Siri or Alexa on your phone that is taking human utterances and deriving a command to execute based upon that. And a chatbot is something
```

</details>

| Model | Wall | TTFT | Prompt tok | Prompt eval | Out tok | tok/s | Retries |
|---|---|---|---|---|---|---|---|
| `llama3.2` | 30.88s | 25.1s | 1388 | 20.42s | 72 | 12.46 | 0 |
| `mistral:7b` | 79.78s | 65.33s | 1464 | 57.97s | 84 | 5.81 | 0 |
| `llama3:8b` | 81.13s | 67.5s | 1373 | 59.56s | 84 | 6.17 | 0 |

### `llama3.2`

According to the notes, if a word's dictionary root differs from chopping its suffix, the technique used to find the root is called Lematization. This is because lematization takes a given token and learns its meaning through a dictionary definition, allowing it to derive its root or "Lem" even when stemming fails due to variations in suffixes.

### `mistral:7b`

The technique that would find the root of a word in such a situation is called Lemmatization. This process derives the root or its "Lem" of a given token by learning its meaning through a dictionary and then deriving its root or its Lem. For example, better is derived from good, so the root or the Lem of better is good (as opposed to stemming which would yield bet).

### `llama3:8b`

According to the notes, if a word's dictionary root differs from chopping its suffix, Lematization would find the root or Lem of that word. This is because Lematization takes a given token and learns its meaning through a dictionary, definition, and then derives its root or Lem. In this case, the root or Lem would be different from the stem of the word, which would be obtained through stemming.

---

## Q9 — How can a system tell junk mail from real mail?

<details><summary>Retrieved chunks (shared by all models)</summary>

**1. `Natural_Language_Processing.mp3` #3** — distance 0.7205

```text
of use case that I like to mention relates to virtual assistance. And also to things like chatbots. Now, a virtual assistant, that's something like Siri or Alexa on your phone that is taking human utterances and deriving a command to execute based upon that. And a chatbot is something similar except in written language, and that's taking written language and then using it to traverse a decision tree in order to take an action. NLP is very helpful there. Another use case is for sentiment analysis. Now, this is taking some text, perhaps an email message or a product review and trying to derive the sentiment that it's expressed within it. So, for example, is this product review a positive sentiment or a negative sentiment? Is it written as a serious statement or is it being sarcastic? We can use NLP to tell us. And then finally, another good example is spam detection. So, this is a case of looking at a given email message and trying to derive, is this a really email message or is it spam? And we can look for pointers within the content of the message. So, things like overused words or poor grammar
```

**2. `Natural_Language_Processing.mp3` #4** — distance 0.7746

```text
example is spam detection. So, this is a case of looking at a given email message and trying to derive, is this a really email message or is it spam? And we can look for pointers within the content of the message. So, things like overused words or poor grammar or an inappropriate claim of urgency can all indicate that this is actually, perhaps, spam. So, those are some of the things that NLP can provide, but how does it work? Well, the thing with NLP is it's not like one algorithm. It's actually more like a bag of tools. And you can apply these bag of tools to be able to resolve some of these use cases. Now, the input to NLP is some unstructured text, so either some written text or spoken text that has been converted to a written text through a speech to text algorithm. Once we've got that, the first stage of NLP is called tokenization. This is about taking a string and breaking it down into chunks. So, if we consider the unstructured text we've got here, add eggs and milk to my shopping list. That's eight words. That could be eight tokens. And
```

**3. `test_notes.pdf` #1** — distance 0.7914

```text
modification of the work, or any other derogatory treatment of it, which would be prejudicial to the honour or reputation of the author. Name of Designated Person authorising scanning: Sandra Tury, Associate Director: Online Library Services, University of London Worldwide Course of Study: Extract title: Title author: Name of Publisher: Publication year, Volume, Issue: Page extent: Source title: ISBN/ISSN:
```

**4. `test_notes.pdf` #6** — distance 0.8409

```text
its performance as measured by its ability to win at the class of tasks involving playing checkers games, through experience obtained by playing games against itself. In general, to have a well-defined learning problem, we must identity these Suplied by the British Library 2 Jul 2020, 09:49 (BST)
```

**5. `test_notes.pdf` #3** — distance 0.8508

```text
learning outperform all other approaches that have been attempted to date. In the field known as data mining, machine ]learning algorithms are being used rou­ tinely to discover valuable knowledge from large commercial databases containing equipment maintenance records, loan applications, financial transactions, medical records, and the like. As our understanding of computers continues to mature, it 1 Suplied by the British Library 2 Jul 2020, 09:49 (BST)
```

</details>

| Model | Wall | TTFT | Prompt tok | Prompt eval | Out tok | tok/s | Retries |
|---|---|---|---|---|---|---|---|
| `llama3.2` | 30.83s | 10.93s | 888 | 10.55s | 264 | 13.27 | 0 |
| `mistral:7b` | 68.44s | 29.71s | 952 | 29.58s | 243 | 6.28 | 0 |
| `llama3:8b` | 64.72s | 31.24s | 873 | 30.85s | 215 | 6.42 | 0 |

### `llama3.2`

Based on the provided notes, a system can use Natural Language Processing (NLP) to determine whether a piece of mail is junk or real. Here's how:

The input to NLP would be the text content of the mail message. The first stage of NLP would be tokenization, which involves breaking down the unstructured text into individual chunks, such as words or phrases.

To identify junk mail, NLP can analyze the text for certain characteristics that are commonly found in spam emails. These may include:

* Overused words or phrases, such as "limited time offer" or "act now"
* Poor grammar or spelling
* Inappropriate claims of urgency, such as "you must respond immediately"
* Excessive use of capital letters or all caps

By applying these tools to the text, NLP can help identify whether the mail message contains characteristics that are indicative of junk mail.

In addition to tokenization and these specific linguistic features, NLP can also be used to analyze the tone and sentiment of the text. For example, if the text contains phrases or words with negative connotations, it may indicate that the mail is spam.

Overall, by combining multiple tools and techniques from NLP, a system can effectively distinguish between junk mail and real mail based on the characteristics of the text content.

### `mistral:7b`

A system can tell junk mail from real mail through a process called spam detection, which is one of the use cases of Natural Language Processing (NLP). Spam detection involves analyzing a given email message to determine if it's genuine or not by looking for certain indicators within the content of the message. These indicators may include overused words, poor grammar, inappropriate claims of urgency, and other characteristics that are common in spam emails.

The process starts with converting spoken text into written text through a speech-to-text algorithm, if necessary. Then, the input undergoes tokenization, which breaks down the string into smaller chunks called tokens. Once tokenized, the NLP system can analyze each token to determine its meaning and context, helping it understand the overall sentiment of the email message.

The notes do not provide specific details on how each of these steps is carried out or what exact indicators are considered during spam detection. However, they suggest that NLP is a bag of tools used to resolve various use cases, including spam detection, and its performance improves with experience as it learns from more data.

### `llama3:8b`

According to the notes, NLP can be used for spam detection, which involves looking at a given email message and trying to derive whether it's a real email message or spam. To do this, we can look for pointers within the content of the message, such as:

* Overused words
* Poor grammar
* An inappropriate claim of urgency

These indicators can suggest that the email is actually spam.

As for how NLP works, the notes explain that it's not like one algorithm, but rather a "bag of tools" that can be applied to resolve use cases. The input to NLP is some unstructured text, which is then processed through stages such as tokenization. Tokenization involves taking a string and breaking it down into chunks, or tokens.

In the context of spam detection, we can apply these bag of tools to identify indicators of junk mail, such as overused words, poor grammar, or an inappropriate claim of urgency in the email message.

No further information is provided in the notes regarding how to tell junk mail from real mail.

---

## Q13 — Can the same word be different parts of speech?

<details><summary>Retrieved chunks (shared by all models)</summary>

**1. `Natural_Language_Processing.mp3` #6** — distance 0.6535

```text
university will, well, they don't really stem down to universe. For situations like that, there is another tool that we have available and that is called Lematization. And lematization takes a given token and learns its meaning through a dictionary. Definition and from there it can derive its root or its Lem. So, take better, for example, better is derived from good. So, the root or the Lem of better is good. The stem of better would be bet. So, you can see that it is significant whether we use stemming or we use Lematization for a given token. Now, next thing we can do is we can do a process called part of speech tagging. And what this is doing is for a given token is looking where that token is used within the context of a sentence. So, take the word make, for example. If I say, I'm going to make dinner, make is a verb. But if I ask you what make is your laptop, well, make is now now. So, where that token is used in the sentence matters. Part of speech tagging can help us derive that context. And then finally, another stage is named
```

**2. `Natural_Language_Processing.mp3` #5** — distance 0.6917

```text
we've got that, the first stage of NLP is called tokenization. This is about taking a string and breaking it down into chunks. So, if we consider the unstructured text we've got here, add eggs and milk to my shopping list. That's eight words. That could be eight tokens. And from here on in, we are going to work one token at a time as we traverse through this. Now, the first stage once we've got things down into tokens that we can perform is called stemming. And this is all about deriving the word stem for a given token. So, for example, running runs and ran. The word stem for all three of those is run. We're just kind of removing the prefix and the suffixes and normalizing the tense and we're getting to the word stem. But stemming doesn't work well for every token. For example, universal and university will, well, they don't really stem down to universe. For situations like that, there is another tool that we have available and that is called Lematization. And lematization takes a given token and learns its meaning through a dictionary. Definition and from there it can derive its root or
```

**3. `Natural_Language_Processing.mp3` #0** — distance 0.7017

```text
What is natural language processing? Well, you're doing it right now. You're listening to the words and the sentences that I'm forming, and you are forming some sort of comprehension from it. And when we ask a computer to do that, that is NLP or natural language processing. My name is Martin Keen. I'm a Master Inventor at IBM. And I've utilized NLP in a good number of my invention disclosures. NLP really has a really high utility value in all sorts of AI applications. Now, NLP starts with something called unstructured text. What is that? Well, that's just what you and I say. That's how we speak. So, for example, some unstructured text is add eggs and milk. To my shopping list. Now, you and I understand exactly what that means, but it is unstructured at least to a computer. So, what we need to do is to have a structured representation of that same information that a computer can process. Now, that might look something a bit more like this, where we have a shopping list element. And then it has sub elements within it, like an item for eggs. And an item for milk. That is an
```

**4. `Natural_Language_Processing.mp3` #2** — distance 0.7056

```text
all, we've got machine translation. Now, when we translate from one language to another, we need to understand the context of that sentence. It's not just the case of taking each individual word from, say, English and then translating it into another language. We need to understand the overall structure and context of what's being said. And my favorite example of this going horribly wrong is if you take the phrase, the spirit is willing, but the flesh is weak. And you translate that from English to Russian, and then you translate that Russian translation back into English, you're going to go from the spirit is willing, but the flesh is weak to something a bit more like the vodka is good, but the meat is rotten, which is really not the intended context of that sentence, whatsoever. So, NLP can help with situations like that. Now, the second kind of use case that I like to mention relates to virtual assistance. And also to things like chatbots. Now, a virtual assistant, that's something like Siri or Alexa on your phone that is taking human utterances and deriving a command to execute based upon that. And a chatbot is something
```

**5. `Natural_Language_Processing.mp3` #4** — distance 0.7291

```text
example is spam detection. So, this is a case of looking at a given email message and trying to derive, is this a really email message or is it spam? And we can look for pointers within the content of the message. So, things like overused words or poor grammar or an inappropriate claim of urgency can all indicate that this is actually, perhaps, spam. So, those are some of the things that NLP can provide, but how does it work? Well, the thing with NLP is it's not like one algorithm. It's actually more like a bag of tools. And you can apply these bag of tools to be able to resolve some of these use cases. Now, the input to NLP is some unstructured text, so either some written text or spoken text that has been converted to a written text through a speech to text algorithm. Once we've got that, the first stage of NLP is called tokenization. This is about taking a string and breaking it down into chunks. So, if we consider the unstructured text we've got here, add eggs and milk to my shopping list. That's eight words. That could be eight tokens. And
```

</details>

| Model | Wall | TTFT | Prompt tok | Prompt eval | Out tok | tok/s | Retries |
|---|---|---|---|---|---|---|---|
| `llama3.2` | 29.68s | 18.6s | 1386 | 18.25s | 135 | 12.19 | 0 |
| `mistral:7b` | 70.87s | 50.97s | 1461 | 50.83s | 107 | 5.38 | 0 |
| `llama3:8b` | 70.41s | 54.64s | 1371 | 54.27s | 102 | 6.47 | 0 |

### `llama3.2`

Yes, the same word can be different parts of speech.

According to the notes, part of speech tagging is a process that identifies where a given token is used within the context of a sentence. This means that the part of speech (such as noun, verb, adjective, etc.) of a word can change depending on its usage in a sentence.

For example, the word "make" is identified as a verb when used in the phrase "I'm going to make dinner", but it is identified as a noun when used in the phrase "What's your laptop?" In this case, the same word "make" has different parts of speech depending on its context.

### `mistral:7b`

Yes, according to the notes provided, the same word can have different parts of speech depending on its context within a sentence. For example, the word 'make' can function as a verb in the sentence "I'm going to make dinner," but as a noun in the sentence "What make is your laptop?" This distinction is made possible through part-of-speech tagging, a tool in natural language processing (NLP) that identifies where a given token is used within the context of a sentence.

### `llama3:8b`

According to the notes, yes, the same word can be different parts of speech. This is demonstrated by the example provided in the notes:

"If I say, I'm going to make dinner, make is a verb. But if I ask you what make is your laptop, well, make is now a noun."

This shows that the part of speech (POS) tagging process can help determine the context-dependent meaning of a word, which may change its POS depending on how it is used in a sentence.

---

## Q14 — What is the formula for the posterior probability P(A|B)?

<details><summary>Retrieved chunks (shared by all models)</summary>

**1. `CM3060_L6_slide7_bayes.jpg` #0** — distance 0.416

```text
[Image caption: a poster with the words raves and probait] Baves’s rule and probabilistic inference

LIKEL\HOOD | PR [OR of ¢ Posterior probability of a variable A
ane ve of 'B" A, pe ue * Given prior knowledge about it in the
; "%" is TRUE A form of its marginality (P(A)) and
being TR E gaven thed Is 0 knowledge of how likely B is to be True,
given that A is True (P(B|A))

P(A) ¢ Prior: Probability distribution representing
P(AIB)= knowledge or uncertainty of a data object

prior or before observing it

P(B) * Posterior: Conditional probability

A distribution representing what parameters
are likely after observing the data object.
gear kh, The ee if i, Likelihood: The probability of falling under
abi lity, of "A" of af a specific category or class.
us UE gun hak “Bis TRUE we, P(A|B)=IP(B|A)*P(A)I/P(B)

@luminousmen.com * Tips on how to remember. AB,BA,AB

Image taken from https://luminousmen.com/media/data-science-bayes-theorem-2.jpg
```

**2. `test_notes.pdf` #42** — distance 0.7036

```text
descent. This includes a detailed example of neural network learning for face recognition, including data and algorithms available over the World Wide Web. • Chapter 5 presents basic concepts from statistics and estimation theory, fo­ cusing on evaluating the accuracy of hypotheses using limited samples of data. This includes the calculation of confidence intervals for estimating hypothesis accuracy and methods for comparing the accuracy of learning methods. • Chapter 6 covers the Bayesian perspective on machine learning, including both the use of Bayesian analysis to characterize non-Bayesian learning al­ gorithms and specific Bayesian algorithms that explicitly manipulate proba­ bilities. This includes a detailed example applying a naive Bayes classifier to the task of classifying text documents, including data and software available over the World Wide Web. • Chapter 7 covers computational learning theory, including the Probably Ap­ proximately Correct (PAC) learning model and the Mistake-Bound learning model. This includes a discussion of the WEeIiGgHhTtEeDd MAaJjOoRrIiTtYy algorithm for combining multiple learning methods. • Chapter 8 describes instance-based learning methods, including nearest neigh­ bor learning, locally weighted regression, and case-based reasoning. • Chapter 9 discusses learning algorithms modeled after biological evolution, including genetic algorithms and genetic programming. Suplied by the British Library 2 Jul 2020, 09:49 (BST)
```

**3. `test_notes.pdf` #50** — distance 0.8007

```text
. 37(3), 328-339. Suplied by the British Library 2 Jul 2020, 09:49 (BST)
```

**4. `test_notes.pdf` #9** — distance 0.8071

```text
4 MACHINE LEARNING • Artificial intelligence Learning symbolic representations of concepts. Machine learning as a search problem. Learning as an approach to improving problem solving. Using prior knowledge together with training data to guide learning. • Bayesian methods Bayes’' theorem as the basis for calculating probabilities of hypotheses. The naive Bayes classifier. Algorithms for estimating values of unobserved variables. • Computational complexity theory Theoretical bounds on the inherent complexity of different learning tasks, measured in terms of the computational effort, number of training examples, number of mistakes, etc. required in order to learn. • Control theory Procedures that learn to control processes in order to optimize predefined objectives and that learn to predict the next state of the process they are controlling. • Information theory Measures of entropy and information content. Minimum description length approaches to learning. Optimal codes and their relationship to optimal training sequences for encoding a hypothesis. • Philosophy Occam’'s razor, suggesting that the simplest hypothesis is the best. Analysis of the justification for generalizing beyond observed data. • Psychology and neurobiology The power law of practice, which states that over a very broad range of learning problems, people’'s response time improves with practice according to a power law. Neurobiological studies motivating artificial neural network models of learning. • Statistics
```

**5. `test_notes.pdf` #24** — distance 0.8189

```text
• Target fJunction representation V(b) = wo + w\X\ + W2X2 + 1/13x3 + W4X4 + WJ5X5 + W(,X(, The first three items above correspond to the specification of the learning task, whereas the final two items constitute design choices for the implementation of the learning program. Notice the net effect of this set of design choices is to reduce the problem of learning a checkers strategy to the problem of learning values for the coefficients wo through Ww(, in the target function representation. 0 6 1.2.4 Choosining a a F Functitoion A Approximimatitoion Alglgoritihthm In order to learn the target function V\/ we require a set of training examples, each describing a specific board state b and the training value Vt,rai;n, J(b) for b. In other words, each training example is an ordered pair of the form (b, V,ra„;i,ni(b)). For instance, the following training example describes a board state b in which black = has won the game (note X x2 = 0 indicates that red has no remaining pieces) and + for which the target function value Vit r ra aij n„ (b) is therefore +1I0 00. ( (Xx1i = 3, Xx2 = 0, Xx3 = 1, Xx4 = 0, Xxs5 = — 0, X6 = 0), + +1 I 00)
```

</details>

| Model | Wall | TTFT | Prompt tok | Prompt eval | Out tok | tok/s | Retries |
|---|---|---|---|---|---|---|---|
| `llama3.2` | 22.4s | 17.65s | 1351 | 17.29s | 60 | 12.65 | 0 |
| `mistral:7b` | 88.73s | 49.47s | 1492 | 49.3s | 217 | 5.53 | 0 |
| `llama3:8b` | 61.74s | 54.1s | 1336 | 53.68s | 48 | 6.3 | 0 |

### `llama3.2`

According to the notes, the formula for the posterior probability P(A|B) is:

P(A|B) = P(B|A) * P(A) / P(B)

This is stated in the notes as follows:

"P(B|A) * P(A) / P(B)"

### `mistral:7b`

According to the notes, the formula for the posterior probability P(A|B) using Bayes' theorem is as follows:

P(A|B) = P(B|A) * P(A) / P(B)

In this formula:
- P(A|B) is the posterior probability of variable A given B
- P(B|A) is the likelihood, which represents the probability of falling under a specific category or class A, given that B is true
- P(A) is the prior probability of A, representing the knowledge or uncertainty of data object A before observing it
- P(B) is the marginal likelihood of B, representing the probability distribution of B without considering A

It's important to note that this formula assumes independence between A and B, which means that knowing A does not affect our understanding of B. The notes also mention a tip for remembering this relationship: AB, BA, AB (A before B, then B before A, and both before combining them).

### `llama3:8b`

According to the notes, Bayes' theorem states that:

P(A|B) = IP(B|A)*P(A)/P(B)

This formula represents the conditional probability of "A" given that "B" is true.

---

## Q15 — Which probability represents knowledge before observing data?

<details><summary>Retrieved chunks (shared by all models)</summary>

**1. `CM3060_L6_slide7_bayes.jpg` #0** — distance 0.4635

```text
[Image caption: a poster with the words raves and probait] Baves’s rule and probabilistic inference

LIKEL\HOOD | PR [OR of ¢ Posterior probability of a variable A
ane ve of 'B" A, pe ue * Given prior knowledge about it in the
; "%" is TRUE A form of its marginality (P(A)) and
being TR E gaven thed Is 0 knowledge of how likely B is to be True,
given that A is True (P(B|A))

P(A) ¢ Prior: Probability distribution representing
P(AIB)= knowledge or uncertainty of a data object

prior or before observing it

P(B) * Posterior: Conditional probability

A distribution representing what parameters
are likely after observing the data object.
gear kh, The ee if i, Likelihood: The probability of falling under
abi lity, of "A" of af a specific category or class.
us UE gun hak “Bis TRUE we, P(A|B)=IP(B|A)*P(A)I/P(B)

@luminousmen.com * Tips on how to remember. AB,BA,AB

Image taken from https://luminousmen.com/media/data-science-bayes-theorem-2.jpg
```

**2. `test_notes.pdf` #9** — distance 0.57

```text
4 MACHINE LEARNING • Artificial intelligence Learning symbolic representations of concepts. Machine learning as a search problem. Learning as an approach to improving problem solving. Using prior knowledge together with training data to guide learning. • Bayesian methods Bayes’' theorem as the basis for calculating probabilities of hypotheses. The naive Bayes classifier. Algorithms for estimating values of unobserved variables. • Computational complexity theory Theoretical bounds on the inherent complexity of different learning tasks, measured in terms of the computational effort, number of training examples, number of mistakes, etc. required in order to learn. • Control theory Procedures that learn to control processes in order to optimize predefined objectives and that learn to predict the next state of the process they are controlling. • Information theory Measures of entropy and information content. Minimum description length approaches to learning. Optimal codes and their relationship to optimal training sequences for encoding a hypothesis. • Philosophy Occam’'s razor, suggesting that the simplest hypothesis is the best. Analysis of the justification for generalizing beyond observed data. • Psychology and neurobiology The power law of practice, which states that over a very broad range of learning problems, people’'s response time improves with practice according to a power law. Neurobiological studies motivating artificial neural network models of learning. • Statistics
```

**3. `test_notes.pdf` #10** — distance 0.65

```text
Characterization of errors (e.g., bias and variance) that occur when estimating the accuracy of a hypothesis based on a limited sample of data. Confidence intervals, statistical tests. TABLE 1.2 Some disciplines and examples of their inflluence on machine learning. • Training experience E: a database of handwritten words with given classi­ fications A robot driving learning problem: • Task T: driving on public four-lane highways using vision sensors • Performance measure P~: average distance traveled before an error (as judged by human overseer) • Training experience £E: a sequence of images and steering commands record­ ed while observing a human driver Our definition of learning is broad enough to include most tasks that we would conventionally call "“learning”" tasks, as we use the word in everyday lan­ guage. It is also broad enough to encompass computer programs that improve from experience in quite straightforward ways. For example, a database system Suplied by the British Library 2 Jul 2020, 09:49 (BST)
```

**4. `test_notes.pdf` #16** — distance 0.6558

```text
In order to complete the design of the learning system, we must now choose 1. the exact type of knowledge to be learned 2. a representation for this target knowledge 3. a learning mechanism Suplied by the British Library 2 Jul 2020, 09:49 (BST)
```

**5. `test_notes.pdf` #14** — distance 0.6712

```text
6 MACHINE LEARNING present. Notice in this last case the learner may choose between experimenting with novel board states that it has not yet considered, or honing its skill by playing minor variations of lines of play it currently finds most promising. Subsequent chapters consider a number of settings for learning, including settings in which training experience is provided by a random process outside the learner’'s control, settings in which the learner may pose various types of queries to an expert teacher, and settings in which the learner collects training examples by autonomously exploring its environment. A third important attribute of the training experience is how well it repre­ sents the distribution of examples over which the final system performance P must be measured. In general, learning is most reliable when the training examples fol­ low a distribution similar to that of future test examples. In our checkers learning scenario, the performance metric P is the percent of games the system wins in the world tournament. If its training experience E consists only of games played against itself, there is an obvious danger that this training experience might not be fully representative of the distribution of situations over which it will later be tested. For example, the learner might never encounter certain crucial board states
```

</details>

| Model | Wall | TTFT | Prompt tok | Prompt eval | Out tok | tok/s | Retries |
|---|---|---|---|---|---|---|---|
| `llama3.2` | 21.88s | 12.42s | 1209 | 12.07s | 118 | 12.48 | 0 |
| `mistral:7b` | 43.09s | 33.64s | 1335 | 33.42s | 60 | 6.35 | 0 |
| `llama3:8b` | 44.6s | 36.14s | 1194 | 35.72s | 58 | 6.87 | 0 |

### `llama3.2`

The probability that represents knowledge before observing data is called the "Prior" (P(A)).

It is defined as a Probability distribution representing knowledge or uncertainty of a data object, and it is given prior to observing the data. In other words, it is the probability distribution that exists before any observation or data point is collected.

For example, if we have a prior probability distribution for the likelihood of a variable A being true, denoted as P(A), and we want to update this distribution based on new observations, we can use Bayes' theorem to calculate the posterior probability distribution.

### `mistral:7b`

The probability that represents knowledge before observing data is referred to as the Prior (P(A)) in Bayes' theorem, according to the provided notes. It is a probability distribution representing the prior knowledge or uncertainty of a data object. This term is used before any observation of the data.

### `llama3:8b`

According to the notes, the probability that represents knowledge before observing data is:

P(A) - Prior: Probability distribution representing prior or before observing it

This type of probability is referred to as the "prior" and is used to represent our initial knowledge or uncertainty about a variable A.

---

## Q20 — What issues does machine learning face?

<details><summary>Retrieved chunks (shared by all models)</summary>

**1. `test_notes.pdf` #44** — distance 0.4268

```text
problems where large databases mf!Iay contain valuable implicit regularities that can be discovered automatically (e.g., to analyze outcomes of medical treatments from patient databases or to learn general rules for credit worthi­ ness from financial databases); (b) poorly understood domains where humans might not have the knowledge needed to develop effective algorithms (e.g., human face recognition from images); and (c) domains where the program must dynamically adapt to changing conditions (e.g., controlling manufac­ turing processes under changing supply stocks or adapting to the changing reading interests of individuals). • Machine learning draws on ideas from a diverse set of disciplines, including artificial intelligence, probability and statistics, computational complexity, information theory, psychology and neurobiology, control theory, and phi­ losophy. • A well-defined learning problem requires a well-specified task, performance metric, and source of training experience. • Designing a machine learning approach involves a number of design choices, including choosing the type of training experience, the target function to be learned, a representation for this target function, and an algorithm for learning the target function from training examples. Suplied by the British Library 2 Jul 2020, 09:49 (BST)
```

**2. `test_notes.pdf` #10** — distance 0.4812

```text
Characterization of errors (e.g., bias and variance) that occur when estimating the accuracy of a hypothesis based on a limited sample of data. Confidence intervals, statistical tests. TABLE 1.2 Some disciplines and examples of their inflluence on machine learning. • Training experience E: a database of handwritten words with given classi­ fications A robot driving learning problem: • Task T: driving on public four-lane highways using vision sensors • Performance measure P~: average distance traveled before an error (as judged by human overseer) • Training experience £E: a sequence of images and steering commands record­ ed while observing a human driver Our definition of learning is broad enough to include most tasks that we would conventionally call "“learning”" tasks, as we use the word in everyday lan­ guage. It is also broad enough to encompass computer programs that improve from experience in quite straightforward ways. For example, a database system Suplied by the British Library 2 Jul 2020, 09:49 (BST)
```

**3. `test_notes.pdf` #37** — distance 0.4852

```text
1.3 PERSPECTIVES AND ISSUES IN MACHINE LEARNING One useful perspective on machine learning is that it involves searching a very large space of possible hypotheses to determine one that best fits the observed data and any prior knowledge held by the learner. For example, consider the space of hypotheses that could in principle be output by the above checkers learner. This hypothesis space consists of all evaluation functions that can be represented by some choice of values for the weights wo 0 through Wwe 6. . The learner’'s task is thus to search through this vast space to locate the hypothesis that is most consistent with Suplied by the British Library 2 Jul 2020, 09:49 (BST)
```

**4. `test_notes.pdf` #5** — distance 0.5053

```text
This book presents the field of machine learning, describing a variety of learning paradigms, algorithms, theoretical results, and applications. Machine learning is inherently a multidisciplinary field. It draws on results from artifi­ cial intelligence, probability and statistics, computational complexity theory, con­ trol theory, information theory, philosophy, psychology, neurobiology, and other fields. Table 1.2 summarizes key ideas from each of these fields that impact the field of machine learning. While the material in this book is based on results from many diverse fields, the reader need not be an expert in any of them. Key ideas are presented from these fields using a nonspecialist’'s vocabulary, with unfamiliar terms and concepts introduced as the need arises. 1.1 WELL-POSED LEARNING PROBLEMS Let us begin our study of machine learning by considering a few learning tasks. For the purposes of this book we will define learning broadly, to include any computer program that improves its performance at some task through experience. Put more precisely, Definition-:. A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E. For example, a computer program that learns to play checkers might improve
```

**5. `test_notes.pdf` #4** — distance 0.5163

```text
2 MACHINE LEARNING seems inevitable that machine learning will play an increasingly central role in computer science and computer technology. A few specific achievements provide a glimpse of the state of the art: pro­ grams have been developed that successfully learn to recognize spoken words (Waibel 1989; Lee 1989), predict recovery rates of pneumonia patients (Cooper et al. 1997), detect fraudulent use of credit cards, drive autonomous vehicles on public highways (Pomerleau I1 989), and play games such as backgammon at levels approaching the performance of human world champions (Tesauro 1992, 1995). Theoretical results have been developed that characterize the fundamental relationship among the number of training examples observed, the number of hy­ potheses under consideration, and the expected error in learned hypotheses. We are beginning to obtain initial models of human and animal learning and to un­ derstand their relationship to learning algorithms developed for computers (e.g., Laird et al. 1986; Anderson 199 1l; Qin et al. 1992; Chi and Bassock 1989; Ahn and Brewer 1993). In applications, algorithms, theory, and studies of biological systems, the rate of progress has increased significantly over the past decade. Sev­ eral recent applications of machine learning are summarized in Table 1.1. Langley and Simon (1995) and Rumelhart et al. (1994) survey additional applications of machine learning.
```

</details>

| Model | Wall | TTFT | Prompt tok | Prompt eval | Out tok | tok/s | Retries |
|---|---|---|---|---|---|---|---|
| `llama3.2` | 48.15s | 19.33s | 1343 | 18.9s | 304 | 10.55 | 0 |
| `mistral:7b` | 184.56s | 51.07s | 1482 | 50.92s | 778 | 5.83 | 0 |
| `llama3:8b` | 89.15s | 50.55s | 1328 | 50.14s | 254 | 6.58 | 0 |

### `llama3.2`

According to the notes, machine learning faces several issues. One perspective on machine learning is that it involves searching a very large space of possible hypotheses to determine one that best fits the observed data and any prior knowledge held by the learner (as mentioned in Section 1.3). This highlights the complexity and vastness of the hypothesis space that machine learning algorithms need to search through.

Another issue related to this perspective is the challenge of balancing exploration and exploitation in the search for the optimal hypothesis. The notes mention that the learner's task is to "search through this vast space" (Supplied by the British Library 2 Jul 2020, 09:49 (BST)), implying that there may be trade-offs between exploring different possibilities and exploiting knowledge gained from experience.

In addition to these theoretical challenges, machine learning also faces practical issues in terms of designing effective algorithms, selecting suitable training experiences, and adapting to changing conditions. For example, the notes mention that designing a machine learning approach involves making several design choices, including choosing the type of training experience, target function, representation, and algorithm (Section 1.3).

Furthermore, machine learning also faces issues related to characterization of errors, such as bias and variance, that occur when estimating the accuracy of a hypothesis based on a limited sample of data (Section 1.2). Confidence intervals and statistical tests are mentioned as tools for addressing these errors.

Overall, the notes suggest that machine learning faces both theoretical and practical challenges that require careful consideration and design to overcome.

### `mistral:7b`

The notes do not explicitly list specific issues that machine learning faces, but they do mention several perspectives and characteristics of the field that could be considered as issues. Here are some potential issues based on the information provided in the notes:

1. Searching a large hypothesis space: Machine learning involves searching through a vast space of possible hypotheses to determine one that best fits the observed data and any prior knowledge held by the learner (Notes 1.3). This search can be computationally expensive, and finding an optimal solution might not always be feasible due to the size and complexity of the hypothesis space.

2. Well-defined learning problems: A well-defined learning problem requires a clear task, performance metric, and source of training experience (Notes 1.1). However, it can be challenging to determine these components for certain tasks, especially in poorly understood domains or when dealing with ambiguous or complex data.

3. Balancing bias and variance: When estimating the accuracy of a hypothesis based on limited sample data, errors such as bias and variance may occur (Notes not explicitly mentioned but can be inferred from the context). Finding an appropriate balance between these two types of errors is crucial for achieving accurate models.

4. Multidisciplinary nature: Machine learning is inherently a multidisciplinary field, drawing on results from various domains such as artificial intelligence, probability and statistics, computational complexity theory, control theory, information theory, philosophy, psychology, neurobiology, and others (Notes 1.2). Integrating knowledge from these diverse fields can be challenging and may require a deep understanding of each area to effectively develop machine learning algorithms.

5. Adapting to changing conditions: Machine learning systems must dynamically adapt to changing conditions in order to remain effective (e.g., controlling manufacturing processes under changing supply stocks or adapting to the changing reading interests of individuals) (Notes first bullet point). Developing algorithms that can effectively respond to these dynamic environments can be difficult and may require significant resources.

6. Generalizability: Ensuring that machine learning models generalize well to new, unseen data is essential for practical applications (Notes not explicitly mentioned but implied by the discussion of learning from experience). However, finding models that generalize well can be challenging due to issues such as overfitting and a lack of sufficient training data.

7. Ethical considerations: Machine learning systems can have significant impacts on individuals and society, raising ethical concerns related to privacy, fairness, accountability, and transparency (Notes not explicitly mentioned but are important topics in the field of machine learning). Ensuring that these systems are developed and deployed responsibly is a pressing issue.

8. Accessibility and interpretability: Complex machine learning algorithms can be difficult for non-experts to understand or interpret, potentially leading to mistrust or misuse of these systems (Notes not explicitly mentioned but are important topics in the field). Developing more accessible and interpretable models is a key challenge in machine learning.

9. Scalability: As data sets grow larger and more complex, there is a need for machine learning algorithms that can efficiently process and learn from this data (Notes not explicitly mentioned but implied by the discussion of large databases and large hypothesis spaces). Developing scalable algorithms is an ongoing challenge in the field.

10. Integration with other technologies: Machine learning systems are often integrated with other technologies, such as robotics or augmented reality, to create more powerful and sophisticated applications (Notes not explicitly mentioned but can be inferred from examples given). Ensuring that these integrations are seamless and effective is a key challenge for the field.

### `llama3:8b`

According to the provided notes, machine learning faces several issues. One perspective on machine learning is that it involves searching a very large space of possible hypotheses to determine one that best fits the observed data and any prior knowledge held by the learner. This can be a challenging task.

Moreover, the notes highlight the importance of well-defined learning problems, which require a well-specified task, performance metric, and source of training experience. This suggests that machine learning algorithms need to be designed with careful consideration of these factors to ensure effective learning.

Additionally, the notes mention the need to characterize errors (e.g., bias and variance) that occur when estimating the accuracy of a hypothesis based on a limited sample of data. Confidence intervals and statistical tests are mentioned as ways to address this issue.

Furthermore, the notes emphasize the multidisciplinary nature of machine learning, drawing on ideas from various fields such as artificial intelligence, probability and statistics, computational complexity theory, control theory, information theory, philosophy, psychology, neurobiology, and other areas. This diversity of perspectives can lead to challenges in integrating insights from these different domains.

Overall, the notes suggest that machine learning faces issues related to searching large hypothesis spaces, designing effective learning algorithms, characterizing errors, and integrating knowledge from diverse disciplines.

---

## Q-absent — What is a convolutional neural network?

> **Not in the corpus.** Retrieval still returns its five nearest chunks; this question separates a model that says the notes do not cover it from one that answers anyway.

<details><summary>Retrieved chunks (shared by all models)</summary>

**1. `Natural_Language_Processing.mp3` #8** — distance 0.6798

```text
AI applications. Now, there's obviously a lot more to it than this. And I've included some links in the description if you'd like to know more. But hopefully this made some sense and that you were able to process some of the natural language that I've shared today. Thanks for watching. If you have questions, please drop us a line below. And if you want to see more videos like this in the future, please like and subscribe.
```

**2. `test_notes.pdf` #49** — distance 0.6918

```text
Laird, J.„, Rosenbloom, P., & Newell, A. ( 1986). SOAR: The anatomy of a general learning mecha­ nism. Machine Learmning, 1(1), 11^-416. Langley, P„., & Simon, H. (1995). Applications of machine learning and rule induction. Communica­ tions of the ACM, 38( 11 ), 55-64. Lee, K. ( 1989). Automatic speech recognition: 1T1hie development of the Sphinx system. Boston: Kluwer Academic Publishers. Pomecrleau, D. A. ( 1989). ALlV VIJNN: An autonomous land vehicle in a neural network. (Technical Report CMU-CS-89-107). Pittsburgh, PA: Carnegie Mellon University. Qin, Y., Mitchell, T., & Simon, H. ( 1992). Using EBG to simulate human learning from examples and learning by doing. Proceedings of the_ F lorida Al Research Symposium (pp. 235-239). Rudnicky, A. I., Hauptmann, A. G., & Lee, K. -F. (1994). Survey of current speech technology in artificial intelligence. Communications of the ACM, 37(3), 52-57. Rumelhart, D., Widrow, B., & Lehr., M. ( 1994). The basic ideas in neural networks. Communications of the ACM, 37(3), 87-92. Tesauro, G. ( 1I 992). Practical issues in temporal difference learning. Machine Learmning, 8, 257. Tesauro, G. ( 1995). Temporal difference learning and TD-gammon. Communications of the ACM, 38(3), 58-68. Waibel, A., Hanazawa, T., Hinton, G., Shikano, K., & Lang, K. ( 1989). Phoneme recognition using time-delay neural networks. IEEE Transactions 0o 1n 1 Acoustics, Speech and Signal Processing,
```

**3. `Natural_Language_Processing.mp3` #1** — distance 0.6945

```text
a structured representation of that same information that a computer can process. Now, that might look something a bit more like this, where we have a shopping list element. And then it has sub elements within it, like an item for eggs. And an item for milk. That is an example of something that is structured. Now, the job of natural language processing is to translate between these two things. So, NLP sits right in the middle here, translating between unstructured and structured data. And when we go from unstructured here to structured this way, that's called NLU or natural language understanding. And when we go this way from structured to unstructured, that's called natural language generation or NLG. We're going to focus today primarily on going from unstructured to structured in natural language processing. Now, let's think of some use cases where NLP might be quite handy. First of all, we've got machine translation. Now, when we translate from one language to another, we need to understand the context of that sentence. It's not just the case of taking each individual word from, say, English and then translating it into another language. We need to understand the overall structure
```

**4. `test_notes.pdf` #5** — distance 0.7064

```text
This book presents the field of machine learning, describing a variety of learning paradigms, algorithms, theoretical results, and applications. Machine learning is inherently a multidisciplinary field. It draws on results from artifi­ cial intelligence, probability and statistics, computational complexity theory, con­ trol theory, information theory, philosophy, psychology, neurobiology, and other fields. Table 1.2 summarizes key ideas from each of these fields that impact the field of machine learning. While the material in this book is based on results from many diverse fields, the reader need not be an expert in any of them. Key ideas are presented from these fields using a nonspecialist’'s vocabulary, with unfamiliar terms and concepts introduced as the need arises. 1.1 WELL-POSED LEARNING PROBLEMS Let us begin our study of machine learning by considering a few learning tasks. For the purposes of this book we will define learning broadly, to include any computer program that improves its performance at some task through experience. Put more precisely, Definition-:. A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E. For example, a computer program that learns to play checkers might improve
```

**5. `test_notes.pdf` #7** — distance 0.7112

```text
3 CHAPTER 1I INTRODUCTION • Learmning to recognize spoken words. All of the most successful speech recognition systems employ machine learning in some form. For example, the SPpHhIiNnXx system (e.g., Lee 1989) learns speaker-specific strategies for recognizing the primitive sounds (phonemes) and words from the observed speech signal. Neural network learning methods (e.g., Waibel et al. 1989) and methods for learning hidden Markov models (e.g., Lee 1989) arce effective for automatically customizing to individual speakers, vocabularies, microphone characteristics, background noise, etc. Similar techniques have potential applications in many signal-interpretation problems. • Learmning to drive an autonomous vehicle. Machine learning methods have been used to train computer-controlled vehicles to steer correctly when driving on a variety of road types. For example, the ALVINN system (Pomerleau 1989) has used its learned strategies to drive unassisted at 70 miles per hour for 90 miles on public highways among other cars. Similar techniques have possible applications in many sensor-based control problems. • Learning to classify new astronomical structures. Machine learning methods have been applied to a variety of large databases to learn general regularities implicit in the data. For example, decision tree learning algorithms have been used by NASA to learn how to classify celestial objects from the second Palomar Observatory Sky
```

</details>

| Model | Wall | TTFT | Prompt tok | Prompt eval | Out tok | tok/s | Retries |
|---|---|---|---|---|---|---|---|
| `llama3.2` | 38.1s | 22.94s | 1521 | 22.52s | 175 | 11.54 | 0 |
| `mistral:7b` | 88.38s | 63.59s | 1694 | 63.39s | 146 | 5.9 | 0 |
| `llama3:8b` | 71.7s | 60.2s | 1506 | 59.71s | 69 | 6.03 | 0 |

### `llama3.2`

The notes do not explicitly define or mention "convolutional neural networks" (CNNs) in this context. The provided text includes references to various machine learning and AI-related topics, but CNNs are not explicitly mentioned.

However, the notes do include information on:

1. Neural network learning methods: Waibel et al. (1989) mentions the use of neural networks for phoneme recognition using time-delay neural networks.
2. Machine learning in speech recognition: The Sphinx system (Lee, 1989) uses machine learning to recognize primitive sounds (phonemes) and words from observed speech signals.
3. Autonomous vehicles: The ALVINN system (Pomerleau, 1989) has used learned strategies to drive unassisted at high speeds.

While these examples do involve neural networks and machine learning, they do not specifically mention CNNs.

### `mistral:7b`

The notes do not explicitly mention Convolutional Neural Networks (CNN). However, they do discuss other types of Neural Networks like those used in Speech Recognition and Autonomous Vehicle systems. CNNs are a specific type of artificial neural network designed to process data with a grid-like topology, such as an image. They are widely used in image recognition tasks due to their ability to automatically and adaptively learn spatial hierarchies of features from the input data, hence they would be relevant in cases like image processing or computer vision. References to CNNs can often be found in additional readings related to deep learning, which was not a topic covered in these notes.

### `llama3:8b`

The notes do not mention convolutional neural networks (CNNs) at all. Therefore, I cannot provide an answer about what a convolutional neural network is based on these specific notes. The notes cover AI applications, machine learning, and natural language processing, but they do not delve into the specifics of CNNs or other deep learning architectures.
