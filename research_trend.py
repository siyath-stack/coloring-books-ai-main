import os
import json
import random
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_MODEL = (
    os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    or DEFAULT_GEMINI_MODEL
)
HISTORY_FILE = "used_topics.json"
DAILY_PLAN_FILE = "daily_book_plan.json"

class ScenePlan(BaseModel):
    action: str = Field(description="A distinct action that advances the character's story on this page")
    setting: str = Field(default="", description="A simple setting with a few story-relevant objects or scenery details")
    composition: str = Field(default="", description="A page layout or viewpoint that varies from nearby scenes")
    micro_prompt: str = Field(description="A short, child-friendly story caption unique to this scene")

class DailyBookPlan(BaseModel):
    subject: str = Field(description="Unique character subject name and species (e.g. 'cheerful baby sea otter')")
    category: str = Field(description="Niche category (e.g. 'Freshwater Lagoon Explorers')")
    target_age: str = Field(description="Target age bracket (e.g. 'Ages 2-5 Toddler Bold & Easy')")
    character_dna: str = Field(description="Strict anatomical definition anchor enforcing species accuracy (e.g. 'adorable chubby baby sea otter, round squishy belly, small ears, simple whiskers, no human features, 2D vector line art')")
    habitat_rules: str = Field(description="Habitat and background constraints")
    scenes: list[ScenePlan] = Field(min_length=30, max_length=30, description="Exactly 30 sequential scenes telling a morning-to-night story arc")


def has_distinct_story_scenes(scenes):
    """Return whether every scene has unique story and visual details."""
    for field in ("action", "setting", "composition", "micro_prompt"):
        values = [getattr(scene, field, "").strip().casefold() for scene in scenes]
        if len(set(values)) != len(values):
            return False
    return True


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def generate_fallback_plan(history):
    print("[Notice] Gemini API unavailable or key missing. Activating fallback plan...")

    fallback_templates = [
        {
            "subject": "Barnaby the Baby Capybara",
            "category": "Barnaby's Big Day",
            "target_age": "Ages 2-5 Toddler Bold & Easy",
            "character_dna": "adorable chubby baby capybara pup, round squishy body, tiny rounded ears, blunt snout, short legs, no tail, 2D vector line art",
            "habitat_rules": "Simple open scenes with a few large story details and plenty of white space",
        },
        {
            "subject": "Hamish the Baby Highland Calf",
            "category": "Hamish's Big Day",
            "target_age": "Ages 2-5 Toddler Bold & Easy",
            "character_dna": "adorable chubby baby highland cow calf, smooth round body, tiny horn nubs, big round snout, short legs, 2D vector line art",
            "habitat_rules": "Simple open scenes with a few large story details and plenty of white space",
        },
        {
            "subject": "Pip the Playful Sea Otter Pup",
            "category": "Pip's Big Day",
            "target_age": "Ages 2-5 Toddler Bold & Easy",
            "character_dna": "adorable chubby baby sea otter pup, round belly, small rounded ears, simple whisker dots, 2D vector line art",
            "habitat_rules": "Simple open coastal scenes around a kelp forest and rocky shore, with a few large story details and plenty of white space",
        }
    ]

    unused = [t for t in fallback_templates if t["subject"] not in history]
    selected = random.choice(unused) if unused else random.choice(fallback_templates)

    story_beats = [
        ("wakes up and stretches beside a tiny seed", "A cozy nest beside the garden gate at sunrise", "A wide view with the animal on the left and the brightening sky above", "A tiny seed is waiting for a big adventure."),
        ("spots the seed and sniffs it with curiosity", "The seed resting on a stepping stone near the nest", "A close view with the animal leaning in from the right", "What could this little seed grow into?"),
        ("carefully nudges the seed into a patch of soft soil", "A small garden bed with a little mound of earth", "A low side view with the animal beside the garden bed", "The seed finds a snug place in the soil."),
        ("pats the soil gently around the seed", "A neat garden bed edged with three round stones", "A front view with the animal behind the garden bed", "A gentle pat helps the seed settle in."),
        ("brings a splash of water to the garden", "A watering can beside the freshly planted seed", "A three-quarter view with water drops arcing toward the soil", "A drink of water helps the seed wake up."),
        ("watches raindrops fall on the garden", "A soft rain cloud above the garden and a small puddle", "A wide view with the animal sheltering beside a leafy plant", "Rain taps a happy rhythm on the garden."),
        ("shares the garden path with a passing snail", "A curved garden path with a snail and tiny mushrooms", "A ground-level view following the winding path", "A slow little visitor joins the garden walk."),
        ("notices a tiny green sprout peeking through", "A new sprout growing in the center of the garden bed", "A close view with the animal looking down at the sprout", "Look! The first green sprout says hello."),
        ("greets a butterfly fluttering above the sprout", "A butterfly, two flowers, and the young sprout", "A tall composition with the butterfly near the top", "A butterfly flutters by to say hello."),
        ("follows the garden path to fetch more water", "A winding path leading from a stream back to the garden", "A side view with the animal walking across the page", "The little gardener sets off for more water."),
        ("fills a leaf-shaped cup at the stream", "A clear stream with smooth stones and reeds", "A wide landscape view with the stream crossing the page", "A sparkling stream fills the little leaf cup."),
        ("carefully carries the water back home", "A garden path lined with daisies and a low fence", "A full-body walking view with flowers on both sides", "Careful steps bring the water safely home."),
        ("pours water beside the growing sprout", "The garden bed with a small cup and curling water drops", "A close side view focused on the animal and the sprout", "The sprout gets another refreshing drink."),
        ("looks up as the sun peeks through the clouds", "Sun rays, a cloud, and the garden fence", "A low-angle view with the animal gazing toward the sky", "Warm sunshine returns to the garden."),
        ("finds a ladybug resting on a broad leaf", "A large leaf with a ladybug and two small flowers", "A close view with the leaf in the foreground", "A spotted friend takes a rest on a leaf."),
        ("carefully steps around the new leaves", "A row of broad leaves growing beside the garden path", "A top-down view of the winding path and leafy row", "The growing leaves make a green little maze."),
        ("watches a flower bud begin to open", "A tall flower bud rising above the garden bed", "A centered portrait view with the animal beside the bud", "A sleepy flower bud starts to yawn open."),
        ("meets a cheerful bird perched on the fence", "A simple fence with a bird and the growing flower", "A wide view with the bird above and the animal below", "A cheerful bird sings a garden song."),
        ("shares a quiet picnic beside the flowers", "A picnic blanket with berries beside the garden", "A slightly overhead view of the blanket and blooming plants", "A berry picnic tastes sweeter with a friend."),
        ("smells the first open flower", "A large open flower with leaves and a small bee", "A close view with the flower in the foreground", "The first flower smells like a sunny day."),
        ("helps a friend find the way through the garden", "A stepping-stone trail between tall but simple flowers", "A winding composition with both animals fully visible", "A kind guide leads a friend along the path."),
        ("watches the flower grow taller than the fence", "A tall leafy flower beside a short garden fence", "A vertical composition with the animal looking upward", "The flower grows taller and taller."),
        ("discovers the flower is a bright sunflower", "A sunflower with a large open center and broad leaves", "A centered view with the sunflower rising behind the animal", "Surprise! The garden has grown a sunflower."),
        ("counts the sunflower's big petals", "A large sunflower with a few simple garden leaves", "A close-up flower view with the animal peeking beside it", "How many sunny petals can you count?"),
        ("welcomes bees buzzing around the sunflower", "A sunflower with two friendly bees and open sky", "A balanced view with bees on opposite sides of the flower", "Busy bees come to visit the sunflower."),
        ("rests in the shade of the tall sunflower", "A large sunflower casting no shadow beside a small bench", "A calm side view with the animal resting under the flower", "A cool resting spot waits beneath the flower."),
        ("shares sunflower seeds with garden friends", "A little pile of seeds with a bird and a rabbit nearby", "A group scene arranged in a gentle arc", "The garden friends share a tasty seed snack."),
        ("watches the evening sky turn starry", "The garden fence beneath a crescent moon and a few stars", "A wide evening view with the animal looking up", "The first stars twinkle above the garden."),
        ("says goodnight to the sunflower", "The sunflower beside the garden gate at twilight", "A quiet close view with the animal beside the flower", "Goodnight, sunflower. Thank you for growing."),
        ("curls up for sleep dreaming of tomorrow's garden", "A cozy nest with a small flower and one safe seed nearby", "A peaceful centered bedtime scene with open white space", "A new seed waits for tomorrow's adventure."),
    ]
    if selected["subject"] == "Pip the Playful Sea Otter Pup":
        story_beats = [
            ("wakes floating in a calm bed of kelp", "A quiet kelp bed on the ocean surface at sunrise", "A wide view with Pip floating near the center and open sky above", "Pip wakes as the morning waves gently roll."),
            ("spots a smooth shell beside the kelp", "A small shell resting on a low tide pool rock", "A close view with Pip looking down at the shell", "A shiny shell catches Pip's curious eye."),
            ("paddles toward a rocky tide pool", "A shallow tide pool among a few rounded shore rocks", "A side view with Pip paddling toward the pool", "Pip sets off to explore the tide pool."),
            ("peeks beneath the clear water", "A simple underwater view with sand and two strands of kelp", "A low viewpoint looking across the pool toward Pip", "What might be hiding under the water?"),
            ("finds a clam tucked into the sand", "A single clam resting in a patch of smooth sand", "A close view with Pip noticing the clam from above", "Pip discovers a clam nestled in the sand."),
            ("brings the clam up to the surface", "A calm patch of ocean beside a low rock", "A three-quarter view of Pip surfacing with the clam", "Up to the surface comes the little clam."),
            ("taps the clam gently with a smooth stone", "A smooth stone and closed clam held above the water", "A close view focused on Pip and the stone", "Tap, tap! Pip uses a smooth stone."),
            ("opens the clam and finds a tasty bite", "An open clam and a few small ripples", "A centered view with Pip floating on its back", "The clam opens, and Pip finds a tasty snack."),
            ("enjoys breakfast while floating", "A calm ocean surface with a few floating kelp leaves", "A relaxed overhead view with Pip floating safely", "Breakfast tastes good on the gentle waves."),
            ("dives along a tall strand of kelp", "Tall kelp fronds rising from the ocean floor", "A vertical composition with Pip swimming beside the kelp", "Pip dives into the swaying kelp forest."),
            ("greets a little fish near the kelp", "Two small fish swimming beside one broad kelp leaf", "A balanced view with Pip and the fish on opposite sides", "A little fish swims by to say hello."),
            ("follows a trail of bubbles", "A few large bubbles rising above the sandy ocean floor", "A diagonal view following bubbles toward Pip", "Pip follows the bubbles on a new adventure."),
            ("notices a sea star resting on a rock", "A single sea star on a broad, flat rock", "A close view with Pip peeking around the rock", "A sea star rests quietly on a sunny rock."),
            ("watches a crab scuttle across the shore", "A small crab crossing a simple stretch of wet sand", "A ground-level view of Pip watching the crab", "A little crab scuttles sideways past Pip."),
            ("swims through a natural kelp arch", "Two curved kelp fronds forming a wide open arch", "A centered view of Pip swimming through the arch", "Pip glides through a swaying kelp arch."),
            ("rests beside a sunlit shore rock", "A broad smooth rock beside a few calm waves", "A quiet side view with Pip resting by the rock", "A warm rock makes a cozy place to rest."),
            ("spots a cluster of mussels", "A few large mussels attached to one shore rock", "A close view with Pip looking at the mussels", "Pip finds mussels gathered on a rock."),
            ("splashes playfully in the shallows", "A shallow shoreline with a few simple ripples", "A wide view with Pip making a small splash", "Pip makes a playful splash in the shallows."),
            ("meets another otter pup near the kelp", "Two sea otter pups floating beside a kelp frond", "A full-body view with both pups comfortably spaced", "A new otter friend joins Pip's ocean day."),
            ("plays a gentle game of peekaboo", "A broad kelp leaf between two floating otter pups", "A playful view with each pup peeking around the leaf", "Peekaboo! Pip and a friend giggle in the kelp."),
            ("explores a sandy stretch of shore", "A simple sandy shore with a few round shells", "A wide landscape view with Pip walking along the waterline", "Pip finds a sandy path beside the waves."),
            ("carries a tiny shell back to the water", "One small shell beside a trail of soft footprints", "A low side view of Pip carrying the shell", "Pip carries a little shell back to the sea."),
            ("watches a wave curl around the rocks", "A small curling wave between two rounded rocks", "A view from behind Pip looking toward the wave", "A gentle wave curls around the shore rocks."),
            ("floats on its back beneath the open sky", "A calm ocean surface with a few distant clouds", "An overhead view of Pip floating with plenty of open space", "Pip drifts gently beneath the open sky."),
            ("shares a clam snack with its otter friend", "Two otter pups floating with one opened clam between them", "A friendly centered view with both pups fully visible", "Pip shares a tasty clam with a friend."),
            ("paddles home as the sun begins to set", "The kelp bed beneath a simple evening sky", "A wide view with Pip paddling toward the kelp", "The evening sun colors the waves as Pip heads home."),
            ("settles into the kelp bed for a rest", "A quiet floating kelp bed on the calm ocean", "A peaceful close view with Pip nestled among kelp leaves", "Pip finds a cozy place to rest in the kelp."),
            ("watches the first stars appear", "A calm ocean beneath a crescent moon and a few stars", "A wide evening view with Pip looking up", "The first stars twinkle above the sea."),
            ("says goodnight to the moonlit waves", "A few gentle waves beneath a bright crescent moon", "A quiet view with Pip beside a low kelp frond", "Goodnight, gentle waves. Goodnight, moon."),
            ("curls up in the kelp and dreams of tomorrow", "A cozy kelp bed floating under the night sky", "A peaceful centered bedtime scene with open white space", "Pip drifts to sleep, dreaming of another ocean day."),
        ]
    scenes = [
        {
            "action": action,
            "setting": setting,
            "composition": composition,
            "micro_prompt": caption,
        }
        for action, setting, composition, caption in story_beats
    ]

    return {
        "subject": selected["subject"],
        "category": selected["category"],
        "target_age": selected["target_age"],
        "character_dna": selected["character_dna"],
        "habitat_rules": selected["habitat_rules"],
        "scenes": scenes
    }

def generate_daily_plan():
    history = load_history()
    recent_history_str = ", ".join(history[-40:]) if history else "None"
    plan_data = None

    if GEMINI_API_KEY and HAS_GENAI:
        prompt = f"""
Create a simple 30-page toddler coloring-book story with clean, bold black-and-white line art.

TASK:
1. Choose a cute animal subject not in this recent list: [{recent_history_str}].
2. Write exactly 30 sequential scenes that tell one simple morning-to-night story with a clear beginning, discoveries, and ending.
3. Keep the same animal recognizable in every scene, but make every page visually distinct: change its action, setting, a few story-relevant objects, and composition or viewpoint.
4. Give each scene a short, child-friendly story caption that is specific to that page and is not repeated.
5. Keep scenes uncluttered and easy for toddlers to color: one prominent animal, only a few large simple supporting details, plenty of open white space, and no cropped subjects.
6. Require smooth bold black outlines, large open coloring areas, and a pure white background. No gray, shading, shadows, gradients, textures, filled regions, or rendered words in the illustration.
"""
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DailyBookPlan,
                    temperature=0.8,
                ),
            )
            if response and (response.parsed or response.text):
                parsed = response.parsed or json.loads(response.text)
                plan_data = DailyBookPlan.model_validate(parsed).model_dump()
        except Exception as error:
            error_text = str(error).upper()
            if "503" in error_text or "UNAVAILABLE" in error_text:
                print("[Notice] Gemini returned 503; using the local fallback immediately.")
            else:
                print(f"[Notice] Gemini plan generation failed; using local fallback: {error}")

    if not plan_data:
        plan_data = generate_fallback_plan(history)

    validated_plan = DailyBookPlan.model_validate(plan_data)
    if not has_distinct_story_scenes(validated_plan.scenes):
        print("[Notice] Repeated story details found; using the distinct local fallback plan.")
        validated_plan = DailyBookPlan.model_validate(generate_fallback_plan(history))
    with open(DAILY_PLAN_FILE, "w", encoding="utf-8") as f:
        json.dump(validated_plan.model_dump(), f, indent=2)

    history.append(validated_plan.subject)
    save_history(history)

    print("=====================================================")
    print(f"Selected Subject: '{validated_plan.subject}'")
    print(f"Category: {validated_plan.category} ({validated_plan.target_age})")
    print(f"Character DNA: {validated_plan.character_dna}")
    print(f"Total Scenes Generated: {len(validated_plan.scenes)}")
    print(f"Saved to '{DAILY_PLAN_FILE}'.")
    print("=====================================================")

if __name__ == "__main__":
    generate_daily_plan()
