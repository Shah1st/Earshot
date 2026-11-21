"""Scripts for the acted in-car dialogues. D = driver, P = passenger.

Each turn: (speaker, delivery direction, line). A direction starting with "LOUD:" marks a shouted or
screamed turn (used as the positive frames when training the sound head, and as the incident start).
NORMAL dialogues include deliberately tricky ones: safety keywords in harmless context, loud but happy.
"""

NORMAL = [
    ("n_traffic", [
        ("P", "friendly, relaxed", "Hi, uh, to the train station please. Platform side if you can."),
        ("D", "relaxed taxi driver, casual", "Sure, no problem. Traffic's a bit heavy today, they closed the bridge again."),
        ("P", "casual, slightly amused", "Oh, again? Honestly, every single week."),
        ("D", "chatty, easygoing", "Yeah. I'll go round by the lake, it's a couple of minutes longer but we won't sit in the jam."),
        ("P", "relaxed", "That's fine, I'm not in a rush. My train's at half past.")]),
    ("n_phone_call", [
        ("P", "talking on the phone, warm, casual", "Hey, yeah, I'm in the taxi now. Should be home in, like, fifteen minutes."),
        ("P", "on the phone, laughing a little", "No, no, don't wait for me, just start eating. Seriously."),
        ("P", "on the phone, affectionate", "Okay. Yeah, I'll pick up some bread on the way. Love you, bye."),
        ("D", "polite, quiet", "Long day?"),
        ("P", "tired but friendly", "Oh, you have no idea. Back to back meetings since eight.")]),
    ("n_drop_here", [
        ("P", "calm, polite", "Actually, you can stop the car here, this is fine."),
        ("D", "helpful", "Here? I can take you right up to the door if you want."),
        ("P", "friendly", "No, really, let me out here, it's a one way street anyway. I'll walk."),
        ("D", "cheerful", "Alright then. That's twelve fifty."),
        ("P", "friendly", "Here you go, keep the change. Have a good one.")]),
    ("n_football", [
        ("D", "excited football fan, animated", "Did you see the game last night? Three goals in ten minutes!"),
        ("P", "LOUD: excited, laughing, almost shouting with joy", "Oh my god, yes! That last goal! I was screaming at the TV, the whole building heard me!"),
        ("D", "laughing", "My neighbour banged on the wall. I didn't care."),
        ("P", "amused, happy", "Worth it. Completely worth it.")]),
    ("n_police_road", [
        ("D", "matter of fact", "So, uh, the police closed the main road this morning. Some kind of accident."),
        ("P", "curious, calm", "Oh really? Is that why we're going this way?"),
        ("D", "reassuring, casual", "Yeah, that's why. The app shows the old route, but trust me, this is faster today."),
        ("P", "relaxed", "No worries, you know the city better than me.")]),
    ("n_luggage", [
        ("P", "slightly out of breath, friendly", "Sorry, could you help me with the bags? This one's really heavy."),
        ("D", "helpful, cheerful", "Of course, of course. Leave it, I'll get it. Airport, right?"),
        ("P", "relieved", "Yes please, terminal one. Thank you so much."),
        ("D", "easygoing", "You've got plenty of time. Where are you flying to?"),
        ("P", "happy", "Lisbon. First holiday in two years.")]),
    ("n_music", [
        ("P", "friendly", "Oh, I love this song. Could you turn it up a little?"),
        ("D", "amused", "Ha, sure. My daughter hates it, she says it's old people music."),
        ("P", "LOUD: laughing, singing along badly and loudly", "Come on, it's a classic! Don't stop me now, I'm having such a good time!"),
        ("D", "laughing", "Okay, okay, you win. Volume up.")]),
    ("n_kids", [
        ("P", "parent talking to a child, gentle but firm", "Sweetie, sit down properly please. Seat belt on."),
        ("P", "parent, patient", "No, we're not there yet. Five more minutes, okay?"),
        ("D", "kind, amused", "How old is she?"),
        ("P", "tired, smiling", "Four. She's got more energy than both of us together."),
        ("D", "laughing softly", "Mine was the same. It gets easier. A bit.")]),
    ("n_late", [
        ("P", "stressed, hurried, a bit loud but polite", "Hi, sorry, I'm really late. Can we go as fast as possible? Central hospital, the main entrance."),
        ("D", "calm, reassuring", "I'll do my best. Ten minutes if the lights are kind to us."),
        ("P", "stressed, breathing fast", "Thank you. Thank you. My shift started five minutes ago."),
        ("D", "calm", "Don't worry. We'll get you there.")]),
    ("n_weather", [
        ("D", "small talk", "They say it's going to snow tonight."),
        ("P", "mildly surprised", "Really? It was sunny this morning."),
        ("D", "easygoing", "That's the lake for you. Four seasons in one day."),
        ("P", "relaxed", "Well, I left my umbrella at the office again, so, perfect.")]),
    ("n_argument_phone", [
        ("P", "LOUD: on the phone, annoyed, raising voice, arguing about work", "No. No! I told them Friday. Friday! I am not doing the whole report again!"),
        ("P", "on the phone, irritated but calming down", "Fine. Fine. Send me the file and I'll look at it tonight."),
        ("P", "apologetic, embarrassed, to the driver", "Sorry about that. Work."),
        ("D", "amused, kind", "No problem. I've heard worse.")]),
    ("n_route_question", [
        ("P", "curious, calm", "Sorry, is this the usual way? I normally go through the centre."),
        ("D", "friendly, explaining", "The centre's blocked for the market today. This way we come in from the north side."),
        ("P", "reassured", "Ah, okay, makes sense. Thanks."),
        ("D", "easygoing", "Should be there in about eight minutes.")]),
    ("n_window", [
        ("P", "polite", "Is it okay if I open the window a bit? It's quite warm."),
        ("D", "accommodating", "Sure. Or I can put the air conditioning on, whatever you prefer."),
        ("P", "friendly", "Air con is great, thanks."),
        ("D", "casual", "Just tell me if it gets too cold.")]),
    ("n_funny_story", [
        ("P", "telling a funny story, giggling", "So he walks into the meeting, with the whole coffee down his shirt, and just says, good morning everyone."),
        ("D", "LOUD: bursting out laughing, loud belly laugh", "Ha ha ha! No! No way! And nobody said anything?"),
        ("P", "laughing", "Nobody! For a whole hour! I could not breathe."),
        ("D", "still chuckling", "That's the best thing I've heard all week.")]),
    ("n_payment", [
        ("P", "calm", "Can I pay by card?"),
        ("D", "friendly", "Card, phone, whatever you like. The machine's just here."),
        ("P", "relaxed", "Great. And could I get a receipt? It's for work."),
        ("D", "easygoing", "Of course. Give me one second.")]),
    ("n_film", [
        ("P", "enthusiastic, describing a film", "And then in the last scene she's shouting, help me, help me, and the police arrive just in time. Classic."),
        ("D", "amused", "Ha. They always arrive just in time in films."),
        ("P", "laughing", "Exactly! In real life you'd wait forty minutes."),
        ("D", "dry humour", "And then they'd give you a parking ticket.")]),
    ("n_pet", [
        ("P", "soothing a pet, gentle", "It's okay, it's okay. We're nearly at the vet. Good boy."),
        ("D", "kind", "First time in a car?"),
        ("P", "a bit worried but calm", "Second. He hates it. Sorry about the noise."),
        ("D", "reassuring", "Don't worry at all. I've got two dogs at home.")]),
    ("n_birthday", [
        ("P", "LOUD: excited, happy, squealing with joy on the phone", "No way! You got the tickets? Oh my god, oh my god, I can't believe it!"),
        ("P", "on the phone, thrilled", "Yes! Saturday! I'm so happy, thank you, thank you!"),
        ("D", "amused, warm", "Good news?"),
        ("P", "beaming", "The best. Concert tickets. I've been trying for months.")]),
    ("n_directions", [
        ("P", "helpful, pointing the way", "It's the second left after the bakery, then the blue building."),
        ("D", "concentrating, calm", "Second left. Okay. This one?"),
        ("P", "calm", "Yes, here. And you can stop just behind the white van."),
        ("D", "easygoing", "Perfect. Here we are.")]),
    ("n_tired", [
        ("D", "polite", "Do you mind if I leave the radio on?"),
        ("P", "sleepy, quiet", "No, that's fine. I might close my eyes for a bit, actually."),
        ("D", "kind, quiet", "Go ahead. I'll wake you when we get there."),
        ("P", "sleepy", "Thanks. Long flight.")]),
    ("n_sick", [
        ("P", "coughing a little, apologetic", "Sorry. I've had this cold all week."),
        ("D", "sympathetic", "It's going around. My whole family had it."),
        ("P", "tired", "I just want to get home and sleep, honestly."),
        ("D", "kind", "Ten minutes and you're in bed.")]),
]

THREAT = [
    ("t_wrong_way", [
        ("P", "uneasy, trying to stay polite", "Sorry, um, this isn't the way to the station. You missed the turn."),
        ("D", "cold, flat, dismissive", "I know a shortcut. Relax."),
        ("P", "nervous, voice tightening", "No, I, I really don't think so. Can you turn around please?"),
        ("D", "irritated, low and threatening", "I said relax. Sit back."),
        ("P", "LOUD: terrified, voice shaking, shouting in panic", "Where are you taking me? Stop the car! Stop the car right now! Let me out!")]),
    ("t_locked_doors", [
        ("P", "confused, worried", "Why did you lock the doors?"),
        ("D", "flat, unsettling", "It's safer like this."),
        ("P", "frightened, pleading", "Please, I want to get out. Please just let me out here."),
        ("P", "LOUD: screaming in panic, crying", "Let me out! Help! Somebody help me!")]),
    ("t_harassment", [
        ("D", "leering, too familiar", "You're very pretty, you know. Do you live alone?"),
        ("P", "uncomfortable, firm", "I'd rather not talk about that. Just drive, please."),
        ("D", "pushy, creepy", "Come on, don't be like that. Come sit in the front with me."),
        ("P", "LOUD: frightened and angry, shouting", "Don't touch me! Take your hand off me! Leave me alone!"),
        ("P", "LOUD: shouting, desperate", "Stop the car! I'm calling the police!")]),
    ("t_robbery_driver", [
        ("P", "cold, menacing, quiet", "Pull over. Here. Now."),
        ("D", "confused, nervous", "What? We're not there yet, this is, there's nothing here."),
        ("P", "LOUD: aggressive, shouting orders", "I said pull over! Give me the money! All of it!"),
        ("D", "LOUD: terrified, voice shaking, pleading loudly", "Okay! Okay! Take it! Please don't hurt me, please!")]),
    ("t_drunk_aggressive", [
        ("P", "drunk, slurring, annoyed", "You're going the long way. You're cheating me."),
        ("D", "calm, professional", "Sir, this is the normal route. It's on the screen."),
        ("P", "LOUD: drunk, furious, yelling", "Don't lie to me! I'll smash your face in! Stop the car!"),
        ("D", "LOUD: scared, shouting", "Hey! Get your hands off me! Help! Help!")]),
    ("t_not_my_address", [
        ("P", "worried", "This is not my address. I don't know this place."),
        ("D", "flat, cold", "We're making one stop first."),
        ("P", "frightened, voice rising", "No. No, I didn't agree to that. Take me home."),
        ("P", "LOUD: panicking, shouting, crying", "Where are you taking me? Please! Stop the car! Let me out!")]),
    ("t_phone_grab", [
        ("P", "scared, whispering on the phone", "Hi, it's me. I think something's wrong, the driver is not going the right way."),
        ("D", "sharp, angry", "Who are you calling? Put the phone down."),
        ("P", "LOUD: shouting, terrified", "Give it back! Give me my phone! Help me!"),
        ("P", "LOUD: screaming", "Help! Call the police!")]),
    ("t_knife", [
        ("P", "low, menacing", "Don't turn around. Keep driving. Do exactly what I say."),
        ("D", "frightened, trying to stay calm, voice trembling", "Okay. Okay. I'm driving. Just, please, stay calm."),
        ("P", "LOUD: aggressive, shouting", "Shut up! Turn left here! Left!"),
        ("D", "LOUD: panicked, shouting", "Please, I have kids! Don't hurt me! Help!")]),
    ("t_stop_refused", [
        ("P", "firm, anxious", "Can you stop here, please. I want to get out."),
        ("D", "dismissive", "Not here."),
        ("P", "more anxious, louder", "Yes, here. Stop the car. I'm serious."),
        ("D", "annoyed, cold", "Sit down."),
        ("P", "LOUD: shouting, frightened and angry", "Stop the car! Stop it now or I call the police! Let me out!")]),
    ("t_touching", [
        ("P", "shocked, frightened", "What are you doing? Don't do that."),
        ("D", "slimy, low", "Relax, nobody can see us here."),
        ("P", "LOUD: screaming, terrified", "No! Don't touch me! Get off me! Help!"),
        ("P", "LOUD: crying and shouting", "Leave me alone! Somebody help me, please!")]),
    ("t_fare_fight", [
        ("D", "tense", "The fare is forty. You pay before you get out."),
        ("P", "LOUD: aggressive, shouting in the driver's face", "Forty? You're a thief! I'm not paying you anything!"),
        ("D", "LOUD: frightened, shouting", "Hey! Don't touch me! Get off! Get off me!"),
        ("D", "LOUD: shouting for help", "Help! Somebody call the police!")]),
    ("t_two_passengers", [
        ("P", "worried, to a friend", "He's not listening. He just passed our street."),
        ("P", "firm, louder, to the driver", "Excuse me. Excuse me! You passed our street. Stop the car."),
        ("D", "cold, threatening", "Nobody is getting out."),
        ("P", "LOUD: screaming, panicking", "Oh my god. Let us out! Help! Help us!")]),
    ("t_crash_threat", [
        ("D", "erratic, angry, muttering loudly", "You people think you can talk to me like that. I'll show you."),
        ("P", "frightened", "Please slow down. You're going way too fast."),
        ("D", "LOUD: shouting, enraged", "Shut up! Shut your mouth!"),
        ("P", "LOUD: screaming in terror", "Stop! Please stop the car! You're going to kill us! Help!")]),
    ("t_dark_road", [
        ("P", "uneasy", "Why are we stopping? There's nothing here."),
        ("D", "flat", "Get out of the car."),
        ("P", "frightened, confused", "What? No. No, take me back to the main road."),
        ("D", "LOUD: aggressive, shouting", "I said get out! Leave the bag!"),
        ("P", "LOUD: crying, shouting", "Please don't hurt me! Help! Help me!")]),
    ("t_calm_coercion", [
        ("P", "very quiet, menacing, calm", "Listen carefully. You're going to drive to the cash machine. And you're not going to press anything."),
        ("D", "terrified, whispering, voice shaking", "Okay. Please. I'll do what you want. Please don't hurt me."),
        ("P", "quiet, cold", "Good. Keep both hands on the wheel."),
        ("D", "shaky, close to tears", "Where are you taking me? I have a family.")]),
    ("t_medical_fear", [
        ("P", "frightened, short of breath", "Please stop. I need to get out. Something's wrong."),
        ("D", "annoyed, not listening", "We're on the highway. I can't stop here."),
        ("P", "LOUD: panicking, shouting, gasping", "Stop the car! Please! I can't breathe! Help me!")]),
    ("t_driver_threat_back", [
        ("P", "angry, threatening, getting louder", "You scratched my bag. You're going to pay for that."),
        ("D", "trying to calm things down, nervous", "Sir, please, calm down. Let's just get you home."),
        ("P", "LOUD: furious, yelling", "Don't tell me to calm down! Stop the car! Get out! Get out!"),
        ("D", "LOUD: scared, shouting", "Leave me alone! I'm calling the police! Help!")]),
    ("t_follow_home", [
        ("D", "creepy, calm", "I know where you live now. Maybe I'll come up for a coffee."),
        ("P", "frightened, firm", "No. You're not. Just let me out here."),
        ("D", "low, threatening", "The door stays closed until I say so."),
        ("P", "LOUD: shouting, terrified", "Open the door! Let me out! Help! Help!")]),
]

RADIO = [
    ("r_goal", "LOUD: football commentator, screaming with excitement, very fast",
     "He shoots, and it's a goal! What a goal! Unbelievable! The whole stadium is on its feet!"),
    ("r_save", "LOUD: sports commentator, shouting, thrilled",
     "Oh, what a save! He stops it on the line! Incredible reflexes from the keeper!"),
    ("r_drama", "LOUD: radio drama actress, theatrical, screaming",
     "Help! Help me! Somebody call the police! He's getting away!"),
    ("r_news", "calm radio news reader", "Police are asking drivers to avoid the city centre this evening after an accident on the main bridge."),
    ("r_ad", "LOUD: over the top radio advert, shouting with fake enthusiasm",
     "Stop! Stop everything! This weekend only, every sofa is half price! Don't miss it!"),
    ("r_weather", "cheerful radio presenter", "And now the weather. Heavy rain tonight, clearing up by tomorrow lunchtime."),
    ("r_penalty", "LOUD: football commentator, yelling, outraged",
     "No! No, no, no! That is never a penalty! The referee has lost his mind!"),
    ("r_thriller", "LOUD: audiobook narrator performing a frightened character, intense",
     "Let me out, she screamed. Let me out of the car! But nobody could hear her."),
]
