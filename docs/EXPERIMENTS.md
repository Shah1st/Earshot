# Earshot: what we tested, and what it taught us

Earshot listens for trouble in a taxi ride on the phone itself and only asks an LLM when something happens.
The claim is simple: you do not need an AI model listening to every second of every ride. This page is the story
of how we checked that claim, including the parts that did not go the way we hoped.

## How we measured

We built rides we could score. Each is five minutes of real driving noise recorded inside a car, with harmless
things happening every few seconds (chat, laughter, a cough, the radio) and sometimes a threat: a scream, breaking
glass, or an argument that turns into "let me out". Every sound is placed in the car using acoustics measured in a
real vehicle.

Two things about this data should be said up front. The screams, glass, laughter and noise are real recordings.
The conversations are not: no open dataset of real threats in a car exists, so we wrote the dialogues and had
them performed by a speech model. And the benchmark is small: 24 test rides, 16 incidents, so one incident is
worth about six points.

We kept the test rides apart from everything we tuned on. All numbers below are from those test rides.

## 1. Does an off-the-shelf sound model do the job?

Our first plan was to use YAMNet, a public sound classifier that already knows "Screaming", "Shout" and "Glass".
On its own it opened an event for only 5 of the 16 incidents.

We wanted to know why, so we played the same real screams to it under different conditions. Clean, it recognised
60% of them. With car acoustics, still about the same. With driving noise underneath, 33%: the model starts
hearing "Vehicle" instead. Filtering out the low rumble did not bring it back.

*What we took from it:* the problem is the noise of the car, not the model's knowledge.

## 2. Can a very small trained layer fix that?

We added a tiny layer on top of YAMNet (one multiplication every half second on the phone) and taught it what
distress sounds like in a car. YAMNet itself stays untouched. Events opened went from 5 to 11 of 16.

*What we took from it:* cheap to run, clearly useful, mostly because it hears shouted speech, which YAMNet simply
labels "Speech".

## 3. What do the words add?

A sound alone does not say whether a detour is road works or a kidnapping. So the phone also runs a small offline
speech recogniser and, when something happens, gives the LLM a short text: what was heard, what was said, where
the car is. No audio is sent.

| What the judge gets | Incidents alerted | False alerts per hour | Tokens per hour |
|---|---|---|---|
| Sound labels and GPS only | 5 of 16 | 1.5 | 18,178 |
| Plus the transcript | 9 of 16 | 3.0 | 21,270 |
| Plus waking up on safety phrases ("stop the car") | 11 of 16 | 14.0 | 40,524 |

The transcript nearly doubles what we catch. Waking the judge on phrases added two more incidents but also most
of the false alerts: a radio drama where an actress screams "Help me! Call the police!" is, as text, a real plea.

*What we took from it:* the transcript is the heart of the system. The phrase trigger is a dial between catching
more and crying wolf. We first switched it off; later we turned it back on with a wider list, because for a
safety product a missed threat is worse than a false alarm (see 5).

## 4. What about a scream with no words?

Here we found our most uncomfortable result. The phone hears the scream, but the LLM then throws it away: there
are no words to read, and in car noise a scream is often labelled "Animal" or "Siren".

We tried sending the actual audio clip to an audio LLM instead. It dismissed most of them too.

| How a wordless sound is decided (19 real screams and glass breaks detected) | Kept | Dropped | Tokens |
|---|---|---|---|
| Text judge | 9 | 10 | 692 |
| Audio clip sent to an audio LLM | 7 | 12 | 444 |
| Simple rule: ask the rider if they are fine | 19 | 0 | 0 |

*What we took from it:* for this case the best answer is not an AI call at all. The rule costs nothing, no audio
leaves the phone, and its price is asking the rider after about 7% of loud harmless sounds.

## 5. So, is it cheaper than an AI that hears everything?

We ran the alternative for real: the same model receiving every 30 seconds of every ride.

| System | Incidents alerted | Alerted or rider asked | False alerts per hour | Tokens per hour | Audio leaving the phone |
|---|---|---|---|---|---|
| AI hears the whole ride | 16 of 16 | 16 of 16 | 16.5 | 126,662 | all of it |
| Earshot, as we run it (wide phrase trigger) | 13 of 16 | 16 of 16 | 13.5 | 36,357 | none |
| Earshot, frugal setting (phrase trigger off) | 9 of 16 | 12 of 16 | 2.0 | 14,159 | none |

As we run it: three and a half times fewer tokens, nothing uploaded, and every incident reaches a human (13 as
alerts, 3 as "needs a look"). False alerts are about as frequent as with the always-listening model, and most of
them are one radio drama line. In the frugal setting it is nine times cheaper with eight times fewer false alerts,
and catches less.

Our rides are deliberately busy, with something happening every 8 to 20 seconds. A ride where nothing happens
costs Earshot zero tokens; the always-listening model pays for every second regardless.

*What we took from it:* the saving is real but it is "several times", not "a thousand times". Cost follows
events, not minutes, and one setting moves the system between cheaper and safer.

## 6. Are we fooling ourselves with our own generated voices?

This worried us most. Our small layer had been trained partly on the same synthetic voices it was then tested
on. So we trained a second one on real recordings only (about 3,600 real clips, no synthetic voice at all) and
tested it on the dialogues it had never heard.

On real sounds the two were equal. On the performed dialogues the real-only version caught fewer (6 of 12 against
8), and it found the synthetic "shouting" much less convincing: in one demo scene it scored 0.56 where the
original scored 0.96, and that scene was missed.

So we recorded two scenes ourselves: an abduction with screaming, and a robbery at knifepoint where nobody raises
their voice. Both raised an alert, even with the GPS clues removed. But it was the words that did it: the sound
layer gave our screaming and our quiet scene almost the same score.

*What we took from it:* numbers on performed audio do not tell us how the system behaves on real people, and on
real voices the transcript matters even more than we thought. Two recordings are a hint, not a result.

## 7. Five situations

Short scenes written to cover cases the numbers cannot show:

| Scene | What happens | Earshot |
|---|---|---|
| Forced stop | passenger orders the driver to stop and get out | alert |
| Road works | driver announces a detour, leaves the route, returns | dismissed: "driver explained the route change" |
| Shouting, then a turn | passenger shouts orders, car leaves the route | alert |
| Normal ride | ordinary chat | nothing, zero tokens |
| Engine stall | engine cuts out, driver explains, long stop | dismissed: "minor mechanical issue" |

Across all five, Earshot used 3,405 tokens and the always-listening model 7,705. These scenes were written by us
after the system existed, so they show what it is meant to do, not an independent test. Without the transcript
(experiment 3) scenes two, three and five become indistinguishable.

## What we do not claim

* That it catches everything. A calm threat on a normal route is examined only if the recogniser picks up one of
  our listed phrases through the car noise.
* That the radio is solved. Dramatic speech from the car speakers can still raise a false alert.
* That these numbers hold on real rides. They are measured on a small benchmark with performed conversations.

The code for every experiment is in `ml/scripts/`; the detailed tables are in `docs/ML_STATUS.md`.
