# Evaluation chunk dump — session `eval_set`

Generated 2026-08-09T16:55:21 from 3 source file(s).

Every chunk stored for this session, in full. Written for reading, not for parsing — use it to author evaluation questions against material that is actually retrievable.

| Source file | Type | Chunks |
| --- | --- | --- |
| `CM3060_L6_slide7_bayes.jpg` | image | 1 |
| `Natural_Language_Processing.mp3` | audio | 9 |
| `test_notes.pdf` | pdf | 51 |
| **Total** | | **61** |

---

## `CM3060_L6_slide7_bayes.jpg`

### Chunk 0

- **source_file:** `CM3060_L6_slide7_bayes.jpg`
- **source_type:** image
- **chunk_index:** 0
- **page_or_slide:** _(none)_
- **characters:** 949

````text
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
````

---

## `Natural_Language_Processing.mp3`

### Chunk 0

- **source_file:** `Natural_Language_Processing.mp3`
- **source_type:** audio
- **chunk_index:** 0
- **page_or_slide:** _(none)_
- **characters:** 1087

````text
What is natural language processing? Well, you're doing it right now. You're listening to the words and the sentences that I'm forming, and you are forming some sort of comprehension from it. And when we ask a computer to do that, that is NLP or natural language processing. My name is Martin Keen. I'm a Master Inventor at IBM. And I've utilized NLP in a good number of my invention disclosures. NLP really has a really high utility value in all sorts of AI applications. Now, NLP starts with something called unstructured text. What is that? Well, that's just what you and I say. That's how we speak. So, for example, some unstructured text is add eggs and milk. To my shopping list. Now, you and I understand exactly what that means, but it is unstructured at least to a computer. So, what we need to do is to have a structured representation of that same information that a computer can process. Now, that might look something a bit more like this, where we have a shopping list element. And then it has sub elements within it, like an item for eggs. And an item for milk. That is an
````

### Chunk 1

- **source_file:** `Natural_Language_Processing.mp3`
- **source_type:** audio
- **chunk_index:** 1
- **page_or_slide:** _(none)_
- **characters:** 1202

````text
a structured representation of that same information that a computer can process. Now, that might look something a bit more like this, where we have a shopping list element. And then it has sub elements within it, like an item for eggs. And an item for milk. That is an example of something that is structured. Now, the job of natural language processing is to translate between these two things. So, NLP sits right in the middle here, translating between unstructured and structured data. And when we go from unstructured here to structured this way, that's called NLU or natural language understanding. And when we go this way from structured to unstructured, that's called natural language generation or NLG. We're going to focus today primarily on going from unstructured to structured in natural language processing. Now, let's think of some use cases where NLP might be quite handy. First of all, we've got machine translation. Now, when we translate from one language to another, we need to understand the context of that sentence. It's not just the case of taking each individual word from, say, English and then translating it into another language. We need to understand the overall structure
````

### Chunk 2

- **source_file:** `Natural_Language_Processing.mp3`
- **source_type:** audio
- **chunk_index:** 2
- **page_or_slide:** _(none)_
- **characters:** 1142

````text
all, we've got machine translation. Now, when we translate from one language to another, we need to understand the context of that sentence. It's not just the case of taking each individual word from, say, English and then translating it into another language. We need to understand the overall structure and context of what's being said. And my favorite example of this going horribly wrong is if you take the phrase, the spirit is willing, but the flesh is weak. And you translate that from English to Russian, and then you translate that Russian translation back into English, you're going to go from the spirit is willing, but the flesh is weak to something a bit more like the vodka is good, but the meat is rotten, which is really not the intended context of that sentence, whatsoever. So, NLP can help with situations like that. Now, the second kind of use case that I like to mention relates to virtual assistance. And also to things like chatbots. Now, a virtual assistant, that's something like Siri or Alexa on your phone that is taking human utterances and deriving a command to execute based upon that. And a chatbot is something
````

### Chunk 3

- **source_file:** `Natural_Language_Processing.mp3`
- **source_type:** audio
- **chunk_index:** 3
- **page_or_slide:** _(none)_
- **characters:** 1111

````text
of use case that I like to mention relates to virtual assistance. And also to things like chatbots. Now, a virtual assistant, that's something like Siri or Alexa on your phone that is taking human utterances and deriving a command to execute based upon that. And a chatbot is something similar except in written language, and that's taking written language and then using it to traverse a decision tree in order to take an action. NLP is very helpful there. Another use case is for sentiment analysis. Now, this is taking some text, perhaps an email message or a product review and trying to derive the sentiment that it's expressed within it. So, for example, is this product review a positive sentiment or a negative sentiment? Is it written as a serious statement or is it being sarcastic? We can use NLP to tell us. And then finally, another good example is spam detection. So, this is a case of looking at a given email message and trying to derive, is this a really email message or is it spam? And we can look for pointers within the content of the message. So, things like overused words or poor grammar
````

### Chunk 4

- **source_file:** `Natural_Language_Processing.mp3`
- **source_type:** audio
- **chunk_index:** 4
- **page_or_slide:** _(none)_
- **characters:** 1060

````text
example is spam detection. So, this is a case of looking at a given email message and trying to derive, is this a really email message or is it spam? And we can look for pointers within the content of the message. So, things like overused words or poor grammar or an inappropriate claim of urgency can all indicate that this is actually, perhaps, spam. So, those are some of the things that NLP can provide, but how does it work? Well, the thing with NLP is it's not like one algorithm. It's actually more like a bag of tools. And you can apply these bag of tools to be able to resolve some of these use cases. Now, the input to NLP is some unstructured text, so either some written text or spoken text that has been converted to a written text through a speech to text algorithm. Once we've got that, the first stage of NLP is called tokenization. This is about taking a string and breaking it down into chunks. So, if we consider the unstructured text we've got here, add eggs and milk to my shopping list. That's eight words. That could be eight tokens. And
````

### Chunk 5

- **source_file:** `Natural_Language_Processing.mp3`
- **source_type:** audio
- **chunk_index:** 5
- **page_or_slide:** _(none)_
- **characters:** 1104

````text
we've got that, the first stage of NLP is called tokenization. This is about taking a string and breaking it down into chunks. So, if we consider the unstructured text we've got here, add eggs and milk to my shopping list. That's eight words. That could be eight tokens. And from here on in, we are going to work one token at a time as we traverse through this. Now, the first stage once we've got things down into tokens that we can perform is called stemming. And this is all about deriving the word stem for a given token. So, for example, running runs and ran. The word stem for all three of those is run. We're just kind of removing the prefix and the suffixes and normalizing the tense and we're getting to the word stem. But stemming doesn't work well for every token. For example, universal and university will, well, they don't really stem down to universe. For situations like that, there is another tool that we have available and that is called Lematization. And lematization takes a given token and learns its meaning through a dictionary. Definition and from there it can derive its root or
````

### Chunk 6

- **source_file:** `Natural_Language_Processing.mp3`
- **source_type:** audio
- **chunk_index:** 6
- **page_or_slide:** _(none)_
- **characters:** 1050

````text
university will, well, they don't really stem down to universe. For situations like that, there is another tool that we have available and that is called Lematization. And lematization takes a given token and learns its meaning through a dictionary. Definition and from there it can derive its root or its Lem. So, take better, for example, better is derived from good. So, the root or the Lem of better is good. The stem of better would be bet. So, you can see that it is significant whether we use stemming or we use Lematization for a given token. Now, next thing we can do is we can do a process called part of speech tagging. And what this is doing is for a given token is looking where that token is used within the context of a sentence. So, take the word make, for example. If I say, I'm going to make dinner, make is a verb. But if I ask you what make is your laptop, well, make is now now. So, where that token is used in the sentence matters. Part of speech tagging can help us derive that context. And then finally, another stage is named
````

### Chunk 7

- **source_file:** `Natural_Language_Processing.mp3`
- **source_type:** audio
- **chunk_index:** 7
- **page_or_slide:** _(none)_
- **characters:** 1048

````text
going to make dinner, make is a verb. But if I ask you what make is your laptop, well, make is now now. So, where that token is used in the sentence matters. Part of speech tagging can help us derive that context. And then finally, another stage is named entity recognition. And what this is asking is for a given token is there an entity associated with it. For example, a token of Arizona has an entity of a US state, whereas a token of Ralph has an entity of a person's name. And these are some of the tools that we can apply in this big bag of tools that we have for NLP in order to get from this unstructured human speech through to something structured that a computer can understand. And once we've done that, then we can apply that structured data to all sorts of AI applications. Now, there's obviously a lot more to it than this. And I've included some links in the description if you'd like to know more. But hopefully this made some sense and that you were able to process some of the natural language that I've shared today. Thanks for
````

### Chunk 8

- **source_file:** `Natural_Language_Processing.mp3`
- **source_type:** audio
- **chunk_index:** 8
- **page_or_slide:** _(none)_
- **characters:** 425

````text
AI applications. Now, there's obviously a lot more to it than this. And I've included some links in the description if you'd like to know more. But hopefully this made some sense and that you were able to process some of the natural language that I've shared today. Thanks for watching. If you have questions, please drop us a line below. And if you want to see more videos like this in the future, please like and subscribe.
````

---

## `test_notes.pdf`

### Chunk 0 — page/slide 1

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 0
- **page_or_slide:** 1
- **characters:** 1213

````text
Online library: digital copies Copyright Notice Staff and students of the University of London are reminded that copyright subsists in this extract and the work from which it was taken. This Digital Copy has been made under the terms of a CLA licence which allows Course Users to: • access and download a copy; • print out a copy. This Digital Copy and any digital or printed copy supplied under the terms of this Licence are for use in connection with this Course of Study. They should not be downloaded or printed by anyone other than a student enrolled on the named course. A student enrolled on this course may retain such copies after the end of the course, but strictly for their own personal use only. All copies (including electronic copies) shall include this Copyright Notice and shall be destroyed and/or deleted if and when required by the University. Except as provided for by copyright law, no further copying, storage or distribution (including by e-mail) is permitted without the consent of the copyright holder. The author (which term includes artists and other visual creators) has moral rights in the work and neither staff nor students may cause, or permit, the distortion, mutilation or other
````

### Chunk 1 — page/slide 1

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 1
- **page_or_slide:** 1
- **characters:** 409

````text
modification of the work, or any other derogatory treatment of it, which would be prejudicial to the honour or reputation of the author. Name of Designated Person authorising scanning: Sandra Tury, Associate Director: Online Library Services, University of London Worldwide Course of Study: Extract title: Title author: Name of Publisher: Publication year, Volume, Issue: Page extent: Source title: ISBN/ISSN:
````

### Chunk 2 — page/slide 2

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 2
- **page_or_slide:** 2
- **characters:** 1424

````text
CHAPTER 1 1 INTRODUCTION Ever since computers were invented, we have wondered whether they might be made to learn. If we could understand how to program them to learn—-to improve automatically with experience—-the impact would be dramatic. Imagine comput­ ers learning from medical records which treatments are most effective for new diseases, houses learning from experieienc_ce to optimize energy costs based on the particular usage patterns of their occupants, or personal software assistants learn­ ing the evolving interests of their users in order to highlight especially relevant stories from the online morning newspaper. A successful understanding of how to make computers learn would open up many new uses of computers and new levels of competence and customization. And a detailed understanding of information­ processing algorithms for machine learning might lead to a better understanding of human learning abilities (and disabilities) as well. We do not yet know how to make computers learn nearly as well as people learn. However, algorithms have been invented that are effective for certain types of learning tasks, and a theoretical understanding of learning is beginning to emerge. Many practical computer programs have been developed to exhibit use­ ful types of learning, and significant commercial applications have begun to ap­ pear. For problems such as speech recognition, algorithms based on machine
````

### Chunk 3 — page/slide 2

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 3
- **page_or_slide:** 2
- **characters:** 459

````text
learning outperform all other approaches that have been attempted to date. In the field known as data mining, machine ]learning algorithms are being used rou­ tinely to discover valuable knowledge from large commercial databases containing equipment maintenance records, loan applications, financial transactions, medical records, and the like. As our understanding of computers continues to mature, it 1 Suplied by the British Library 2 Jul 2020, 09:49 (BST)
````

### Chunk 4 — page/slide 3

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 4
- **page_or_slide:** 3
- **characters:** 1425

````text
2 MACHINE LEARNING seems inevitable that machine learning will play an increasingly central role in computer science and computer technology. A few specific achievements provide a glimpse of the state of the art: pro­ grams have been developed that successfully learn to recognize spoken words (Waibel 1989; Lee 1989), predict recovery rates of pneumonia patients (Cooper et al. 1997), detect fraudulent use of credit cards, drive autonomous vehicles on public highways (Pomerleau I1 989), and play games such as backgammon at levels approaching the performance of human world champions (Tesauro 1992, 1995). Theoretical results have been developed that characterize the fundamental relationship among the number of training examples observed, the number of hy­ potheses under consideration, and the expected error in learned hypotheses. We are beginning to obtain initial models of human and animal learning and to un­ derstand their relationship to learning algorithms developed for computers (e.g., Laird et al. 1986; Anderson 199 1l; Qin et al. 1992; Chi and Bassock 1989; Ahn and Brewer 1993). In applications, algorithms, theory, and studies of biological systems, the rate of progress has increased significantly over the past decade. Sev­ eral recent applications of machine learning are summarized in Table 1.1. Langley and Simon (1995) and Rumelhart et al. (1994) survey additional applications of machine learning.
````

### Chunk 5 — page/slide 3

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 5
- **page_or_slide:** 3
- **characters:** 1360

````text
This book presents the field of machine learning, describing a variety of learning paradigms, algorithms, theoretical results, and applications. Machine learning is inherently a multidisciplinary field. It draws on results from artifi­ cial intelligence, probability and statistics, computational complexity theory, con­ trol theory, information theory, philosophy, psychology, neurobiology, and other fields. Table 1.2 summarizes key ideas from each of these fields that impact the field of machine learning. While the material in this book is based on results from many diverse fields, the reader need not be an expert in any of them. Key ideas are presented from these fields using a nonspecialist’'s vocabulary, with unfamiliar terms and concepts introduced as the need arises. 1.1 WELL-POSED LEARNING PROBLEMS Let us begin our study of machine learning by considering a few learning tasks. For the purposes of this book we will define learning broadly, to include any computer program that improves its performance at some task through experience. Put more precisely, Definition-:. A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E. For example, a computer program that learns to play checkers might improve
````

### Chunk 6 — page/slide 3

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 6
- **page_or_slide:** 3
- **characters:** 297

````text
its performance as measured by its ability to win at the class of tasks involving playing checkers games, through experience obtained by playing games against itself. In general, to have a well-defined learning problem, we must identity these Suplied by the British Library 2 Jul 2020, 09:49 (BST)
````

### Chunk 7 — page/slide 4

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 7
- **page_or_slide:** 4
- **characters:** 1448

````text
3 CHAPTER 1I INTRODUCTION • Learmning to recognize spoken words. All of the most successful speech recognition systems employ machine learning in some form. For example, the SPpHhIiNnXx system (e.g., Lee 1989) learns speaker-specific strategies for recognizing the primitive sounds (phonemes) and words from the observed speech signal. Neural network learning methods (e.g., Waibel et al. 1989) and methods for learning hidden Markov models (e.g., Lee 1989) arce effective for automatically customizing to individual speakers, vocabularies, microphone characteristics, background noise, etc. Similar techniques have potential applications in many signal-interpretation problems. • Learmning to drive an autonomous vehicle. Machine learning methods have been used to train computer-controlled vehicles to steer correctly when driving on a variety of road types. For example, the ALVINN system (Pomerleau 1989) has used its learned strategies to drive unassisted at 70 miles per hour for 90 miles on public highways among other cars. Similar techniques have possible applications in many sensor-based control problems. • Learning to classify new astronomical structures. Machine learning methods have been applied to a variety of large databases to learn general regularities implicit in the data. For example, decision tree learning algorithms have been used by NASA to learn how to classify celestial objects from the second Palomar Observatory Sky
````

### Chunk 8 — page/slide 4

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 8
- **page_or_slide:** 4
- **characters:** 1440

````text
Survey (Fayyad et al. 1995). This system is now used to automatically classify all objects in the Sky Survey, which consists of three terrabytes of image data. • Learning to play world-class backgammon. The most successful computer programs for playing games such as backgammon are based on machine learning algorithms. For example, the world’'s top computer program for backgammon, TD-G-GAaMmMmOoNn (Tesauro 1992, 1995), learned its strategy by playing over one million practice games against itself. It now plays at a level competitive with the human world champion. Similar techniques have applications in many practical problems where very large search spaces must be examined efficiently. TABLE 1.1 Some successful applications of machine learning=: three features: the class of tasks, the measure of performance to be improved, and the source of experience. A checkers learning problem: • Task T: playing checkers • Performance measure P: percent of games won against opponents • Training experience E: playing practice games against itself We can specify many learning problems in this fashion, such as learning to recognize handwritten words, or learning to drive a robotic automobile au­ tonomously. A handwriting recognition learning problem: • Task T: recognizing and classifying handwritten words within images • Performance measure P: percent of words correctly classified Suplied by the British Library 2 Jul 2020, 09:49 (BST)
````

### Chunk 9 — page/slide 5

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 9
- **page_or_slide:** 5
- **characters:** 1500

````text
4 MACHINE LEARNING • Artificial intelligence Learning symbolic representations of concepts. Machine learning as a search problem. Learning as an approach to improving problem solving. Using prior knowledge together with training data to guide learning. • Bayesian methods Bayes’' theorem as the basis for calculating probabilities of hypotheses. The naive Bayes classifier. Algorithms for estimating values of unobserved variables. • Computational complexity theory Theoretical bounds on the inherent complexity of different learning tasks, measured in terms of the computational effort, number of training examples, number of mistakes, etc. required in order to learn. • Control theory Procedures that learn to control processes in order to optimize predefined objectives and that learn to predict the next state of the process they are controlling. • Information theory Measures of entropy and information content. Minimum description length approaches to learning. Optimal codes and their relationship to optimal training sequences for encoding a hypothesis. • Philosophy Occam’'s razor, suggesting that the simplest hypothesis is the best. Analysis of the justification for generalizing beyond observed data. • Psychology and neurobiology The power law of practice, which states that over a very broad range of learning problems, people’'s response time improves with practice according to a power law. Neurobiological studies motivating artificial neural network models of learning. • Statistics
````

### Chunk 10 — page/slide 5

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 10
- **page_or_slide:** 5
- **characters:** 1025

````text
Characterization of errors (e.g., bias and variance) that occur when estimating the accuracy of a hypothesis based on a limited sample of data. Confidence intervals, statistical tests. TABLE 1.2 Some disciplines and examples of their inflluence on machine learning. • Training experience E: a database of handwritten words with given classi­ fications A robot driving learning problem: • Task T: driving on public four-lane highways using vision sensors • Performance measure P~: average distance traveled before an error (as judged by human overseer) • Training experience £E: a sequence of images and steering commands record­ ed while observing a human driver Our definition of learning is broad enough to include most tasks that we would conventionally call "“learning”" tasks, as we use the word in everyday lan­ guage. It is also broad enough to encompass computer programs that improve from experience in quite straightforward ways. For example, a database system Suplied by the British Library 2 Jul 2020, 09:49 (BST)
````

### Chunk 11 — page/slide 6

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 11
- **page_or_slide:** 6
- **characters:** 1311

````text
5 CHAPTER 1I INTRODUCTION that allows users to update data entries would fit our definition of a learning system: it improves its performance at answering database queries, based on the experience gained from database updates. Rather than worry about whether this type of activity falls under the usual informal conversational meaning of the word "“learning,”" we wiJlIl simply adopt our technical definition of the class of programs that improve through experience. Within this class we will find many types of problems that require more or less sophisticated solutions. Our concern here is not to analyze the meaning of the English word "“learning”" as it is used in ev­ eryday language. Instead, our goal is to define precisely a class of problems that encompasses interesting forms of learning, to explore algorithms that solve such problems, and to understand the fundamental structure of learning problems and processes. 1.2 DESIGNING A LEARNING SYSTEM In order to illustrate some of the basic design issues and approaches to machine learning, let us consider designing a program to learn to play checkers, with the goal of entering it in the world checkers tournament. We adopt the obvious performance measure: the percent of games it wins in this world tournament. 1.2.1 Choosing the Training Experience
````

### Chunk 12 — page/slide 6

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 12
- **page_or_slide:** 6
- **characters:** 1379

````text
The first design choice we face is to choose the type of training experience from which our system will learn. The type of training experience available can have a significant impact on success or failure of the learner. One key attribute is whether the training experience provides direct or indirect feedback regarding the choices made by the performance system. For example, in learning to play checkers, the system might learn from direct training examples consisting of individual checkers board states and theé correct move for each. Alternatively, it might have available only indirect information consisting of the move sequences and final outcomes of various games played. In this later case, information about the correctness of specific moves early in the game must be inferred indirectly from the fact that the game was eventually won or lost. Here the learner faces an additional problem of credit assignment, or determining the degree to which each move in the sequence deserves credit or blame for the final outcome. Credit assignment can be a particularly difficult problem because the game can be lost even when early moves are optimal, if these are followed later by poor moves. Hence, learning from direct training feedback is typically easier than learning from indirect feedback. A second important attribute of the training experience is the degree to which
````

### Chunk 13 — page/slide 6

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 13
- **page_or_slide:** 6
- **characters:** 555

````text
the learner controls the sequence of training examples. For example, the learner might rely on the teacher to select informative board states and to provide the correct move for each. Alternatively, the learner might itself propose board states that it finds particularly confusing and ask the teacher for the correct move. Or the learner may have complete control over both the board states and (indirect) training classifications, as it does when it learns by playing against itself with no teacher Suplied by the British Library 2 Jul 2020, 09:49 (BST)
````

### Chunk 14 — page/slide 7

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 14
- **page_or_slide:** 7
- **characters:** 1358

````text
6 MACHINE LEARNING present. Notice in this last case the learner may choose between experimenting with novel board states that it has not yet considered, or honing its skill by playing minor variations of lines of play it currently finds most promising. Subsequent chapters consider a number of settings for learning, including settings in which training experience is provided by a random process outside the learner’'s control, settings in which the learner may pose various types of queries to an expert teacher, and settings in which the learner collects training examples by autonomously exploring its environment. A third important attribute of the training experience is how well it repre­ sents the distribution of examples over which the final system performance P must be measured. In general, learning is most reliable when the training examples fol­ low a distribution similar to that of future test examples. In our checkers learning scenario, the performance metric P is the percent of games the system wins in the world tournament. If its training experience E consists only of games played against itself, there is an obvious danger that this training experience might not be fully representative of the distribution of situations over which it will later be tested. For example, the learner might never encounter certain crucial board states
````

### Chunk 15 — page/slide 7

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 15
- **page_or_slide:** 7
- **characters:** 1301

````text
that are very likely to be played by the human checkers champion. In practice, it is often necessary to learn from a distribution of examples that is somewhat different from those on which the final system will be evaluated (e.g., the world checkers champion might not be interested in teaching the program!). Such situ­ ations are problematic because mastery of one distribution of examples will not necessary lead to strong performance over some other distribution. We shall see that most current theory of machine learning rests on the crucial assumption that the distribution of training examples is identical to the distribution of test ex­ amples. Despite our need to make this assumption in order to obtain theoretical results, it is important to keep in mind that this assumption must often be violated in practice. To proceed with our design, let us decide that our system will train by playing games against itself. This has the advantage that no external trainer need be present, and it therefore allows the system to generate as much training data as time permits. We now have a fully specified learning task. A checkers learning problem: • Task T: playing checkers • Performance measure P: percent of games won in the world tournament • Training experience E£: games played against itself
````

### Chunk 16 — page/slide 7

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 16
- **page_or_slide:** 7
- **characters:** 244

````text
In order to complete the design of the learning system, we must now choose 1. the exact type of knowledge to be learned 2. a representation for this target knowledge 3. a learning mechanism Suplied by the British Library 2 Jul 2020, 09:49 (BST)
````

### Chunk 17 — page/slide 8

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 17
- **page_or_slide:** 8
- **characters:** 1217

````text
7 CHAPTER I INTRODUCTION 1.2.2 Choosing the Target Function The next design choice is to determine exactly what type of knowledge will be learned and how this will be used by the performance program. Let us begin with a checkers-playing program that can generate the legal moves from any board state. The program needs only to learn how to choose the best move from among these legal moves. This learning task is representative of a large class of tasks for which the legal moves that define some large search space are known a priori, but for which the best search strategy is not known. Many optimization problems fall into this class, such as the problems of scheduling and controlling manufacturing processes where the available manufacturing steps are well understood, but the best strategy for sequencing them is not. Given this setting where we must learn to choose among the legal moves, the most obvious choice for the type of information to be learned is a program, or function, that chooses the best move for any given board state. Let us call this function ChooseMove and use the notation ChooseMove : B  -> M to indicate that this function accepts as input any board from the set of legal board states B
````

### Chunk 18 — page/slide 8

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 18
- **page_or_slide:** 8
- **characters:** 1195

````text
and produces as output some move from the set of legal moves M. Throughout our discussion of machine learning we wi l1l1 find it useful to reduce the problem of improving performance P at task T to the problem of learning some particu­ lar target function such as Choose Move. The choice of the target function will therefore be a key design choice. Although Clhwoose Move is an obvious choice for the target function in our example, this function will turn out to be very difficult to learn giv_en the kind of in­ direct training experience available to our system. An alternative target functitoion—n­ - and one that will turn out to be easier to learn in this settining—-is an evaluation function that assigns a numerical score to any given board state. Let us call this target function V and again use the notation V : B  -> mi)l to denote that V maps any legal board sta~te from the set B/3 to some real value (we use mil t o denote the set of real numbers). We intend for this target function V to assign higher scores to better board states. If the system can successfully learn such a target function V, then it can easily use it to select the best move from any current board position.
````

### Chunk 19 — page/slide 8

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 19
- **page_or_slide:** 8
- **characters:** 914

````text
This can be accomplished by generating the successor board state produced by every legal move, then using V to choose the best successor state and therefore the best legal move. . . What exactly should be the value of the target function V for any given board state? Of course any evaluation function that assigns higher scores to better board states will do. Nevertheless, we will find it useful to define one particular target function V among the many that produce optimal play. As we shall see, this will make it easier to design a training algorithm. Let us therefore define the target value V (b) for an arbitrary board state b in B, as follows: 1 1I. if bi si sa a final board state that is won, then V(b) = 100 2. if b i si sa a final board state that is lost, then V( (b) = —- 1I0 00 3. if b i si sa a final board state that is drawn, then V( (b) = 0 Suplied by the British Library 2 Jul 2020, 09:49 (BST)
````

### Chunk 20 — page/slide 9

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 20
- **page_or_slide:** 9
- **characters:** 1184

````text
8 MACHINE LEARNING 4. if b is a not a final state in the game, then V (b) = — V (b'), where b'1 is the best final board state that can be achieved starting from b and playing optimally until the end of the game (assuming the opponent plays optimally, as well). While this recursive definition specifies a value of V (b) for every board state b, this definition is not usable by our checkers player because it is not efficiently computable. Except for the trivial cases (cases 1-3) in which the game has already ended, determining the value of V (b) for a particular board state requires (case 4) searching ahead for the optimal line of play, all the way to the end of the game! Because this definition is not efficiently computable by our checkers playing program, we say that it is a nonoperational definition. The goal of learning in this case is to discover an operational description of V; that is, a description that can be used by the checkers-playing program to evaluate states and select moves within realistic time bounds. Thus, we have reduced the learning task in this case to the problem of discovering an operational description of the ideal target fJunction V. It may be
````

### Chunk 21 — page/slide 9

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 21
- **page_or_slide:** 9
- **characters:** 1275

````text
very difficult in general to learn such an operational form of V perfectly. In fact, we often expect learning algorithms to acquire only some approximation to the target function, and for this reason the process of learning the target function is often called fJunction approximation. In the current discussion we will use the symbol V to refer to the function that is actually learned by our program, to distinguish it from the ideal target function V. 1.2.3 Choosisning a a R Represesentatatitoion f ofor t hthe T Target tF Functcitoion Now that we have specified the ideal target function V, we must choose a repre­ sentation that the learning program will use to describe the function V that it will learn. As with earlier design choices, we again have many options. We could, for example, al1low the program to represent V using a large table with a distinct entry specifying the value for each distinct board state. Or we could allow it to represent V using a collection of rules that match against features of the board state, or a quadratic polynomial function of predefined board features, or an arti­ ficial neural network. In general, this choice of representation involves a crucial tradeoff. On one hand, we wish to pick a very expressive representation to allow
````

### Chunk 22 — page/slide 9

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 22
- **page_or_slide:** 9
- **characters:** 689

````text
representing as close an approximation as possible to the ideal target function V. On the other hand, the more expressive the representation, the more training data the program will require in order to choose among the alternative hypotheses it can represent. To keep the discussion brief, let us choose a simple representation: for any given board state, the function V will be calculated as a linear combination of the following board features: • x 1\ : the number of black pieces on the board • Xx2: the number of red pieces on the board • JxC3 3: the number of black kings on the board • xX 4: the number of red kings on the board Suplied by the British Library 2 Jul 2020, 09:49 (BST)
````

### Chunk 23 — page/slide 10

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 23
- **page_or_slide:** 10
- **characters:** 1190

````text
9 CHAPTER 1I INTRODUCTION • x5 5: the number of black pieces threatened by red (i.e., which can be captured on red’'s next turn) • Xx( 6,: : the number of red pieces threatened by black Thus, our learning program will represent V(b) as a linear function of the form V (b) = w o + w1\x1\ + W w2xX 2 + wW 3 xX 3 + wW 4xX 4 + wW 5xX 5 + W( 6,X X( 6, where uw>() through wo are numerical coefficients, or weights, to be chosen by the 0 6 learning algorithm. Learned values for the weights w\ 1 through wo 6 will determine the relative importance of the various board features in determining the value of the board, whereas the weight wo will provide an additive constant to the board value. To summarize our design choices thus far, we have elaborated the original formulation of the learning problem by choosing a type of training experience, a target function to be learned, and a representation for this target function. Our elaborated learning task is now Partial design of a checkers learning program: • Task T: playing checkers • Performance measure P: percent of games won in the world tournament • Training experience E: games played against itself • Target. function: V :.Board - ~» !H
````

### Chunk 24 — page/slide 10

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 24
- **page_or_slide:** 10
- **characters:** 1144

````text
• Target fJunction representation V(b) = wo + w\X\ + W2X2 + 1/13x3 + W4X4 + WJ5X5 + W(,X(, The first three items above correspond to the specification of the learning task, whereas the final two items constitute design choices for the implementation of the learning program. Notice the net effect of this set of design choices is to reduce the problem of learning a checkers strategy to the problem of learning values for the coefficients wo through Ww(, in the target function representation. 0 6 1.2.4 Choosining a a F Functitoion A Approximimatitoion Alglgoritihthm In order to learn the target function V\/ we require a set of training examples, each describing a specific board state b and the training value Vt,rai;n, J(b) for b. In other words, each training example is an ordered pair of the form (b, V,ra„;i,ni(b)). For instance, the following training example describes a board state b in which black = has won the game (note X x2 = 0 indicates that red has no remaining pieces) and + for which the target function value Vit r ra aij n„ (b) is therefore +1I0 00. ( (Xx1i = 3, Xx2 = 0, Xx3 = 1, Xx4 = 0, Xxs5 = — 0, X6 = 0), + +1 I 00)
````

### Chunk 25 — page/slide 10

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 25
- **page_or_slide:** 10
- **characters:** 54

````text
Suplied by the British Library 2 Jul 2020, 09:49 (BST)
````

### Chunk 26 — page/slide 11

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 26
- **page_or_slide:** 11
- **characters:** 1276

````text
10 MACHINE LEARNING Below we describe a procedure that first derives such training examples from the indirect training experience available to the learner, then adjusts the weights w, to best fit these training examples. 1.2.4.1 ESTIMATING TRAINING VALUES Recall that according to our formulation of the learning problem, the only training information available to our learner is whether the game was eventually won or lost. On the other hand, we require training examples that assign specific scores to specific board states. While it is easy to assign a value to board states that correspond to the end of the game, it is less obvious how to assign training values to the more numerous intermediate board states that occur before the game’'s end. Of course the fact that the game was eventually won or lost does not necessarily indicate that every board state along the game path was necessarily good or bad. For example, even if the program looses the game, it may still be the case that board states occurring early in the game should be rated very highly and that the cause of the loss was a subsequent poor move. Despite the ambiguity inherent in estimating training values for intermediate board states, one simple approach has been found to be surprisingly successful.
````

### Chunk 27 — page/slide 11

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 27
- **page_or_slide:** 11
- **characters:** 1228

````text
This approach is to assign the training value of V,rain{(b) for any intermediate board V state b to be V ({Successor ({b)), where V is the learner’'s current approximation to V and where Successor(b) denotes the next board state following hb for which it is again the program’'s turn to move (i.e., the board state following the program’'s move and the opponent’'s response). This rule for estimating training values can be summarized as Rule for estimating training values. V,roaitn,(b) ~«- V(Successor(b)) (I1.I1) While it may seem strange to use the current version of V to estimate training values that will be used to refine this very same function, notice that we are using estimates of the value of the Successor (b) to estimate the value of board state b. In­ V tuitively, we can see this will make sense if V tends to be more accurate for board states closer to game’'s end. In fact, under certain conditions (d discussed in Chap­ ter 13) the approach of iteratively estimating training values based on estimates of successor state values can be proven to converge toward perfect estimates of V,rain„·. 1.2.4.2 ADJUSTING THE WEIGHTS All that remains is to specify the learning algorithm for choosing the weights w, to 1
````

### Chunk 28 — page/slide 11

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 28
- **page_or_slide:** 11
- **characters:** 470

````text
best fit the set of training examples {[{( b, V,rain({b)) }. As a first step we must define 1 what we mean by the best fit to the training data. One common approach is to define the best hypothesis, or set of weights, as that which minimizes the squared V. error E between the training values and the values predicted by the hypothesis V. E = W(Vrrai1n1(bh) ~- V(b)) 2 ({hb, Vi,r, 0a ;i 0n(hb))Ee t/raining examples Suplied by the British Library 2 Jul 2020, 09:49 (BST)
````

### Chunk 29 — page/slide 12

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 29
- **page_or_slide:** 12
- **characters:** 1262

````text
11 CHAPTER I INTRODUCTION Thus, we seek the weights, or equivalently the V, that minimize E for the observed training examples. Chapter 6 discusses settings in which minimizing the sum of squared enr-ors is equivalent to finding the most probable hypothesis given the observed training data. Several algorithms are known for finding weights of a linear function that minimize E defined in this way. In our case, we require an algorithm that will incrementally refine the weights as new training examples become available and that will be robust to errors in these estimated training values. One such algorithm is called the least mean squares, or LMS training rule. For each observed training example it adjusts the weights a small amount in the direction that reduces the error on this training example. As discussed in Chapter 4, this algorithm can be viewed as performing a stochastic gradient-descent search through the space of possible hypotheses (weight values) to minimize the squared error E. The LMS algorithm is defined as follows: LMS weight update rule. For each training example (h/;, V,raw(b)) V1ra1i • Use the current weights to calculate V ( b) • For each weight w;h, update ilt as WlV ;i ^ W ;j +-p Ttj / ( V ,m raun ,(i b) ) - - V(b)) X X[ ; 1
````

### Chunk 30 — page/slide 12

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 30
- **page_or_slide:** 12
- **characters:** 1219

````text
Here 1p1 is a small constant (e.g., 0.1) that moderates the size of the weight update. To get an intuitive understanding for why this weight update rule works, notice that when th~e error ( Vr,rain (b) - V (b)) is zero, no weights are changed. When ('V,,ra;i,„,(b) - V(b)) is positive (i.e., whe_n V(b) is too low), then each weight is 0 increased in proportion to the value of its conr-esponding feature. This will raise the v valauleu eo of fV V(b(b);) ;r erdeudcuicnign g the error. Notice that if the value of some feature x, ;- is zero, then its weight is not altered regardless of the error, so that the only weights updated are those whose features actually occur on the training example board. Surprisingly, in certain settings this simple weight-tuning method can be proven to converge to the least squared error approximation to the V,jra,„n, values (as discussed in Chapter 4 ). 1.2.5 The Final Design The final design of our checkers learning system can be naturally described by four distinct program modules that represent the central components in many learning systems. These four modules, summarized in Figure 1. 1I, are as follows: • The Performance System is the module that must solve the given per­
````

### Chunk 31 — page/slide 12

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 31
- **page_or_slide:** 12
- **characters:** 279

````text
formance task, in this case playing checkers, by using the learned target function(s). It takes an instance of a new problem (new game) as input and produces a trace of its solution (game history) as output. In our case, the Suplied by the British Library 2 Jul 2020, 09:49 (BST)
````

### Chunk 32 — page/slide 13

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 32
- **page_or_slide:** 13
- **characters:** 1284

````text
1}2 MACHINE LEARNING Generator New problem Hypo~1esis (initial game board) ( V) Performance System Training examples Solution trace (game history) {<b,.~mi1 (b,) >, <b2, ~rni1(/J2)>. Critic FIGURE 1l.1l Final design of the checkers learning program. strategy used by the Performance System to select its next move at each step is determined by the learned V evaluation function. Therefore, we expect its performance to improve as this evaluation function becomes increasingly accurate. • The Critic takes as input the history or trace of the game and produces as output a set of training examples of the target function. As shown in the diagram, each training example in this case corresponds to some game state in the trace, along with an estimate V , r a i l„ l of the target function value for this example. In our example, the Critic corresponds to the training rule given by Equation ( 1.1 ). • The Generalizer takes as input the training examples and produces an output hypothesis that is its estimate of the target function. It generalizes from the specific training examples, hypothesizing a general function that covers these examples and other cases beyond the training examples. In our example, the Generalizer corresponds to the LMS algorithm, and the output hypothesis is
````

### Chunk 33 — page/slide 13

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 33
- **page_or_slide:** 13
- **characters:** 582

````text
the function V described by the learned weights wo 0, ... , wW( 6,. • The Experiment Generator takes as input the current hypothesis (currently learned function) and outputs a new problem (i.e., initial board state) for the Performance System to explore. Its role is to pick new practice problems that will maximize the learning rate of the overall system. In our example, the Experiment Generator follows a very simple strategy: It always proposes the same initial game board to begin a new game. More sophisticated strategies Suplied by the British Library 2 Jul 2020, 09:49 (BST)
````

### Chunk 34 — page/slide 14

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 34
- **page_or_slide:** 14
- **characters:** 1291

````text
13 CHAPTER I INTRODUCTION ] could involve creating board positions designed to explore particular regions of the state space. Together, the design choices we made for our checkers program produce specific instantiations for the performance system, critic, generalizer, and experi­ ment generator. Many machine learning systems can be usefully characterized in terms of these four generic modules. The sequence of design choices made for the checkers program is summa­ rized in Figure 1J .2. These design choices have constrained the learning task in a number of ways. We have restricted the type of knowledge that can be acquired to a single linear evaluation function. Furthermore, we have constrained this eval­ uation function to depend on only the six specific board features provided. If the true target function V can indeed be represented by a linear combination of these Determine Type of Training Experience Table of corect Games against self Determine Target Function Board -• move -• value Determine Representation of Learned Function Linear function . Artificial neural of six features network Determine Learning Algorithm programing Completed Design FIGURE 1.2 Summary of choices in designing the checkers learning program. Suplied by the British Library 2 Jul 2020, 09:49 (BST)
````

### Chunk 35 — page/slide 15

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 35
- **page_or_slide:** 15
- **characters:** 1340

````text
14 MACHINE LEARNING particular features, then our program has a good chance to learn it. If not, then the best we can hope for is that it will learn a good approximation, since a program can certainly never learn anything that it cannot at least represent. Let us suppose that a good approximation to the true V function can, in fact, be represented in this form. The question then arises as to whether this learning technique is guaranteed to find one. Chapter 13 provides a theoretical analysis showing that under rather restrictive assumptions, variations on this approach do indeed converge to the desired evaluation function for certain types of search problems. Fortunately, practical experience indicates that this approach to learning evaluation functions is often successful, even outside the range of situations for which such guarantees can be proven. Would the program we have designed be able to learn well enough to beat the human checkers world champion? Probably not. In part, this is because the linear function representation for V is too simple a representation to capture well the nuances of the game. However, given a more sophisticated representation for the target function, this general approach can, in fact, be quite successful. For example, Tesauro ( 1992, 1995) reports a similar design for a program that learns
````

### Chunk 36 — page/slide 15

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 36
- **page_or_slide:** 15
- **characters:** 1399

````text
to play the game of backgammon, by learning a very similar evaluation function over states of the game. His program represents the learned evaluation function using an artificial neural network that considers the complete description of the board state rather than a subset of board features. After training on over one million self-generated training games, his program was able to play very competitively with top-ranked human backgammon players. Of course we could have designed many alternative algorithms for this checkers learning task. One might, for example, simply store the given training examples, then try to find the "“closest”" stored situation to match any new situation (nearest neighbor algorithm, Chapter 8). Or we might generate a large number of candidate checkers programs and allow them to play against each other, keep­ ing only the most successful programs and further elaborating or mutating these in a kind of simulated evolution (genetic algorithms, Chapter 9). Humans seem to follow yet a different approach to learning strategies, in which they analyze, or explain to themselves, the reasons underlying specific successes and failures encountered during play (explanation-based learning, Chapter 11 ). Our design is simply one of many, presented here to ground our discussion of the decisions that must go into designing a learning method for a specific clas3s of tasks.
````

### Chunk 37 — page/slide 15

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 37
- **page_or_slide:** 15
- **characters:** 685

````text
1.3 PERSPECTIVES AND ISSUES IN MACHINE LEARNING One useful perspective on machine learning is that it involves searching a very large space of possible hypotheses to determine one that best fits the observed data and any prior knowledge held by the learner. For example, consider the space of hypotheses that could in principle be output by the above checkers learner. This hypothesis space consists of all evaluation functions that can be represented by some choice of values for the weights wo 0 through Wwe 6. . The learner’'s task is thus to search through this vast space to locate the hypothesis that is most consistent with Suplied by the British Library 2 Jul 2020, 09:49 (BST)
````

### Chunk 38 — page/slide 16

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 38
- **page_or_slide:** 16
- **characters:** 1522

````text
15 CHAPTER I INTRODUCTION the available training examples. The LMS algorithm for fitting weights achieves this goal by iteratively tuning the weights, adding a correction to each weight each time the hypothesized evaluation function predicts a value that differs from the training value. This algorithm works well when the hypothesis representation considered by the learner defines a continuously parameterized space of potential hypotheses. Many of the chapters in this book present algorithms that search a hypothesis space defined by some underlying representation (e.g., linear functions, logical descriptions, decision trees, artificial neural networks). These different hypothesis representations are appropriate for learning different kinds of target functions. For each of these hypothesis representations, the corresponding learning algorithm takes advantage of a different underlying structure to organize the search through the hypothesis space. Throughout this book we will return to this perspective of learning as a search problem in order to characterize learning methods by their search strategies and by the underlying structure of the search spaces they explore. We will also find this viewpoint useful in formally analyzing the relationship between the size of the hypothesis space to be searched, the number of training examples available, and the confidence we can have that a hypothesis consistent with the training data will correctly generalize to unseen examples. 1.3.1 Issues in Machine Learning
````

### Chunk 39 — page/slide 16

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 39
- **page_or_slide:** 16
- **characters:** 1378

````text
Our checkers example raises a number of generic questions about machine learn­ ing. The field of machine learning, and much of this book, is concerned with answering questions such as the following: • What algorithms exist for learning general target functions from specific training examples? JIn what settings will particular algorithms converge to the desired function, given sufficient training data? Which algorithms perform best for which types of problems and representations? • How much training data is sufficient? What general bounds can be found to relate the confidence in learned hypotheses to the amount of training experience and the character of the learner’'s hypothesis space? • When and how can prior knowledge held by the learner guide the process of generalizing from examples? Can prior knowledge be helpful even when it is only approximately correct? • What is the best strategy for choosing a useful next training experience, and how does the choice of this strategy alter the complexity of the learning problem? • What is the best way to reduce the learning task to one or more function approximation problems? Put another way, what specific functions should the system attempt to learn? Can this process itself be automated? • How can the learner automatically alter its representation to improve its ability to represent and learn the target function?
````

### Chunk 40 — page/slide 16

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 40
- **page_or_slide:** 16
- **characters:** 54

````text
Suplied by the British Library 2 Jul 2020, 09:49 (BST)
````

### Chunk 41 — page/slide 17

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 41
- **page_or_slide:** 17
- **characters:** 1421

````text
16 MACHINE LEARNING 1.4 HOW TO READ THIS BOOK This book contains an introduction to the primary algorithms and approaches to machine learning, theoretical results on the feasibility of various learning tasks and the capabilities of specific algorithms, and examples of practical applications of machine learning to real-world problems. Where possible, the chapters have been written to be readable in any sequence. However, some interdependence is unavoidable. If this is being used as a class text, I recommend first covering Chapter I1 and Chapter 2. Following these two chapters, the remaining chapters can be read in nearly any sequence. A one-semester course in machine learning might cover the first seven chapters, followed by whichever additional chapters are of greatest interest to the class. Below is a brief survey of the chapters. • Chapter 2 covers concept learning based on symbolic or logical representa­ tions. It also discusses the general-to-specific ordering over hypotheses, and the need for inductive bias in learning. • Chapter 3 covers decision tree learning and the problem of overfitting the training data. It also examines Occam’'s razor—r-a principle recommending the shortest hypothesis among those consistent with the data. • Chapter 4 covers learning of artificial neural networks, especeially the well-­ studied BAaCckKpPrRo0PpAaGgaAtTiIo0Nn algorithm, and the general approach of gradient
````

### Chunk 42 — page/slide 17

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 42
- **page_or_slide:** 17
- **characters:** 1481

````text
descent. This includes a detailed example of neural network learning for face recognition, including data and algorithms available over the World Wide Web. • Chapter 5 presents basic concepts from statistics and estimation theory, fo­ cusing on evaluating the accuracy of hypotheses using limited samples of data. This includes the calculation of confidence intervals for estimating hypothesis accuracy and methods for comparing the accuracy of learning methods. • Chapter 6 covers the Bayesian perspective on machine learning, including both the use of Bayesian analysis to characterize non-Bayesian learning al­ gorithms and specific Bayesian algorithms that explicitly manipulate proba­ bilities. This includes a detailed example applying a naive Bayes classifier to the task of classifying text documents, including data and software available over the World Wide Web. • Chapter 7 covers computational learning theory, including the Probably Ap­ proximately Correct (PAC) learning model and the Mistake-Bound learning model. This includes a discussion of the WEeIiGgHhTtEeDd MAaJjOoRrIiTtYy algorithm for combining multiple learning methods. • Chapter 8 describes instance-based learning methods, including nearest neigh­ bor learning, locally weighted regression, and case-based reasoning. • Chapter 9 discusses learning algorithms modeled after biological evolution, including genetic algorithms and genetic programming. Suplied by the British Library 2 Jul 2020, 09:49 (BST)
````

### Chunk 43 — page/slide 18

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 43
- **page_or_slide:** 18
- **characters:** 1496

````text
17 CHAPTER I1 INTRODUC.TION • Chapter 1I O0 covers algorithms for learning sets of rules, including Inductive Logic Programming approaches to learning first-order Horn clauses. • Chapter 1I I1 covers explanation-based learning, a learning method that uses prior knowledge to explain observed training examples, then generalizes based on these explanations. • Chapter 12 discusses approaches to combining approximate prior knowledge with available training data in order to improve the accuracy of learned hypotheses. Both symbolic and neural network algorithms are considered. • Chapter 13 discusses reinforcement learnining—-an approach to control learn­ ing that accommodates indirect or delayed feedback as training information. The checkers learning algorithm described earlier in Chapter 1I is a simple example of reinforcement learning. The end of each chapter contains a summary of the main concepts covered, suggestions for further reading, and exercises. Additional updates to chapters, as well as data sets and implementations of algorithms, are available on the World Wide Web at http://www.cs.cmu.edu/"~'tom/mlbook.html. 1.5 SUMMARY AND FURTHER READING Machine learning addresses the question of how to build computer programs that improve their performance at some task through experience. Major points of this chapter include: • Machine learning algorithms have proven to be of great practical value in a variety of application domains. They are especially useful in (a) data mining
````

### Chunk 44 — page/slide 18

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 44
- **page_or_slide:** 18
- **characters:** 1332

````text
problems where large databases mf!Iay contain valuable implicit regularities that can be discovered automatically (e.g., to analyze outcomes of medical treatments from patient databases or to learn general rules for credit worthi­ ness from financial databases); (b) poorly understood domains where humans might not have the knowledge needed to develop effective algorithms (e.g., human face recognition from images); and (c) domains where the program must dynamically adapt to changing conditions (e.g., controlling manufac­ turing processes under changing supply stocks or adapting to the changing reading interests of individuals). • Machine learning draws on ideas from a diverse set of disciplines, including artificial intelligence, probability and statistics, computational complexity, information theory, psychology and neurobiology, control theory, and phi­ losophy. • A well-defined learning problem requires a well-specified task, performance metric, and source of training experience. • Designing a machine learning approach involves a number of design choices, including choosing the type of training experience, the target function to be learned, a representation for this target function, and an algorithm for learning the target function from training examples. Suplied by the British Library 2 Jul 2020, 09:49 (BST)
````

### Chunk 45 — page/slide 19

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 45
- **page_or_slide:** 19
- **characters:** 1526

````text
]18 MACHINE LEARNING • Learning involves search: searching through a space of possible hypotheses to find the hypothesis that best fits the available training examples and other prior constraints or knowledge. Much of this book is organized around dif­ ferent learning methods that search different hypothesis spaces (e.g., spaces containing numerical functions, neural networks, decision trees, symbolic rules) and around theoretical results that characterize conditions under which these search methods converge toward an optimal hypothesis. There are a number of good sources for reading about the latest research results in machine learning. Relevant journals include Machine Learmning, Neural Computation, Neural Networks, Journal of the American Statistical Association, and the IEEE Transactions on Pattern Analysis and Machine Intelligence. There 01 are also numerous annual conferences that cover different aspects of machine learning, including the International Conference on Machine Learning, Neural Information Processing Systems, Conference on Computational Learning The­ ory, International Conference on Genetic Algorithms, International Conference on Knowledge Discovery and Data Mining, European Conference on Machine Learmning, and others. EXERCISES 1.1. Give three computer applications for which machine learning approaches seem ap­ propriate and three for which they seem inappropriate. Pick applications that arec not already mentioned in this chapter,. and include a one-sentence justification for each.
````

### Chunk 46 — page/slide 19

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 46
- **page_or_slide:** 19
- **characters:** 1293

````text
1.2. Pick some learning task not mentioned in this chapter. Describe it informally in a paragraph in English. Now describe it by stating as precisely as possible the task, performance measure, and training experience. Finally, propose a target function to be learned and a target representation. Discuss the main tradeoffs you considered in formulating this learning task. 1.3. Prove that the LMS weight update rule described in this chapter performs a gradient descent to minimize the squared error. In particular, define the squared error E as in the text. Now calculate the derivative of E with respect to the weight w,;-, assuming that V V( hb) is a linear function as defined in the text. Gradient descent is achieved by updating each weight in proportion to -— ,;"'- . Therefore, you must show that the LMS training rule alters weights in this propor;tio6np for each training example it encounters. 1.4. Consider alternative strategies for the Experiment Generator module of Figure 1.2. In particular, consider strategies in which the Experiment Generator suggests new board positions by • Generating random legal board positions • Generating a position by picking a board state from the previous game, then applying one of the moves that was not executed • A strategy of your own design
````

### Chunk 47 — page/slide 19

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 47
- **page_or_slide:** 19
- **characters:** 442

````text
Discuss tradeoffs among these strategies. Which do you feel would work best if the number of training examples was held constant, given the performance measure of winning the most games at the world championships? 1.5. Implement an algorithm similar to that discussed for the checkers problem, but use the simpler game of tic-tac-toe. Represent the learned function V as a linear corin-­ Suplied by the British Library 2 Jul 2020, 09:49 (BST)
````

### Chunk 48 — page/slide 20

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 48
- **page_or_slide:** 20
- **characters:** 1406

````text
19 CHAPTER I INTRODUCTION bination of board features of your choice. To train your program, play it repeatedly against a second copy of the program that uses a fixed evaluation function you cre­ ate by hand. Plot the percent of games won by your system, versus the number of training games played. REFERENCES Ahn, W., & Brewer, W. F. ( 1993). Psychological studies of explanation-based learning. Iln G. DeJong (Ed.), Investigating explanation-based learning. Boston: Kluwer Academic Publishers. Anderson, J. R. ( 1I 991 ). The place of cognitive architecture in rational analysis. In K. VanLehn (Ed.), Architectlulres for intelligence (pp. 1-24). Hillsdale, NJ: Erlbaum. Chi, M. T. H., & Bassock, M. (1989). Learning from examples via self-explanations. In L. Resnick (Ed.), Knowing, learning, and instruction: Essays in honor of Robert Glaser. Hillsdale, NJ: L. Erlbaum Associates. Cooper, G., ect al. ( 1997). An evaluation of machine-learning methods for predicting pneumonia mortality. Artificial Intelligence in Medicine, (to appear). Fayyad, U. M., Uthurusamy, R. (Eds.) (1995). Proceedings of the First International Conference on Knowledge Discovery and Data Mining. Menlo Park, CA: AAAI Press. Fayyad, U. M., Smyth, P., Weir, N., Djorgovski, S. ( 1995). Automated analysis and exploration of image databases: Results, progress, and challenges. Journal {of Intelligent Information Systems, 4, 1-19.
````

### Chunk 49 — page/slide 20

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 49
- **page_or_slide:** 20
- **characters:** 1427

````text
Laird, J.„, Rosenbloom, P., & Newell, A. ( 1986). SOAR: The anatomy of a general learning mecha­ nism. Machine Learmning, 1(1), 11^-416. Langley, P„., & Simon, H. (1995). Applications of machine learning and rule induction. Communica­ tions of the ACM, 38( 11 ), 55-64. Lee, K. ( 1989). Automatic speech recognition: 1T1hie development of the Sphinx system. Boston: Kluwer Academic Publishers. Pomecrleau, D. A. ( 1989). ALlV VIJNN: An autonomous land vehicle in a neural network. (Technical Report CMU-CS-89-107). Pittsburgh, PA: Carnegie Mellon University. Qin, Y., Mitchell, T., & Simon, H. ( 1992). Using EBG to simulate human learning from examples and learning by doing. Proceedings of the_ F lorida Al Research Symposium (pp. 235-239). Rudnicky, A. I., Hauptmann, A. G., & Lee, K. -F. (1994). Survey of current speech technology in artificial intelligence. Communications of the ACM, 37(3), 52-57. Rumelhart, D., Widrow, B., & Lehr., M. ( 1994). The basic ideas in neural networks. Communications of the ACM, 37(3), 87-92. Tesauro, G. ( 1I 992). Practical issues in temporal difference learning. Machine Learmning, 8, 257. Tesauro, G. ( 1995). Temporal difference learning and TD-gammon. Communications of the ACM, 38(3), 58-68. Waibel, A., Hanazawa, T., Hinton, G., Shikano, K., & Lang, K. ( 1989). Phoneme recognition using time-delay neural networks. IEEE Transactions 0o 1n 1 Acoustics, Speech and Signal Processing,
````

### Chunk 50 — page/slide 20

- **source_file:** `test_notes.pdf`
- **source_type:** pdf
- **chunk_index:** 50
- **page_or_slide:** 20
- **characters:** 72

````text
. 37(3), 328-339. Suplied by the British Library 2 Jul 2020, 09:49 (BST)
````
