import os
import random
import string
import time
import traceback  # For detailed error logging
import uuid  # For unique filenames

import matplotlib.font_manager as fm
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from dotenv import load_dotenv
from flask import Flask, request, render_template, send_from_directory, url_for, flash, redirect
from moviepy import ImageSequenceClip  # Use .editor for newer moviepy versions

# --- Mistral AI Integration ---
MISTRAL_AVAILABLE = False
MISTRAL_API_KEY = None
try:
    # --- Mistral AI Integration ---
    from mistralai import UserMessage, SystemMessage, Mistral

    load_dotenv() # Load environment variables from .env file
    MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
    if MISTRAL_API_KEY:
        MISTRAL_AVAILABLE = True
    else:
        print("Warning: MISTRAL_API_KEY not found in environment variables. AI disabled.")
except ImportError:
    UserMessage = None
    Mistral = None
    print("Warning: Mistral AI library not found. AI text generation disabled.")
    print("Install it using: pip install mistralai python-dotenv")
except Exception as _:
    print(f"Warning: Error initializing Mistral AI: {_}. AI disabled.")

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24) # Needed for flash messages (optional but good practice)
app.config['UPLOAD_FOLDER'] = 'output'
app.config['FONT_DIR'] = 'fonts'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 # Limit upload size if adding file uploads later (5MB example)

# Ensure output and font directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FONT_DIR'], exist_ok=True)

# --- Configuration Parameters ---

# Video settings
WIDTH = 1024
HEIGHT = 1024
FPS = 10
DURATION_SECONDS = 5

# Text & Highlighting settings
HIGHLIGHTED_TEXT = "Mother of Dragons"
HIGHLIGHT_COLOR = "yellow"  # Pillow color name or hex code
TEXT_COLOR = "black"
BACKGROUND_COLOR = "white"
FONT_SIZE_RATIO = 0.05  # Adjusted slightly for multi-line potentially
MIN_LINES = 7  # Min number of text lines per frame
MAX_LINES = 10  # Max number of text lines per frame
VERTICAL_SPREAD_FACTOR = 1.5  # Multiplier for line height (1.0 = tight, 1.5 = looser)

# AI Text Generation Settings
AI_GENERATION_ENABLED = MISTRAL_AVAILABLE  # Auto-disable if library missing
# REMOVED: UNIQUE_TEXT_COUNT - we now generate fresh content per frame
MISTRAL_MODEL = "mistral-large-latest"  # Or choose another suitable model
# !! IMPORTANT: Load API Key securely !!
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

# Effect settings
BLUR_TYPE = 'radial'  # Options: 'gaussian', 'radial'
BLUR_RADIUS = 4.0  # Gaussian blur radius, or the radius OUTSIDE which radial blur starts fading strongly
RADIAL_SHARPNESS_RADIUS_FACTOR = 0.3  # For 'radial': Percentage of min(W,H) to keep perfectly sharp around center

# Font settings
FONT_DIR = "fonts"  # Dedicated font folder recommended
MAX_FONT_RETRIES_PER_FRAME = 5
# Generate random words only using ASCII lowercase for fallback/disabled AI
FALLBACK_CHAR_SET = string.ascii_lowercase + " "

# NEW: Add variety and randomness parameters
TEMPERATURE_RANGE = (0.7, 1.2)  # Random temperature range for AI generation
RANDOM_SEED_PER_FRAME = True  # Use different random seed per frame
THEME_VARIETY = True  # Enable theme variety in AI prompts

# --- Helper Functions (Mostly unchanged from original script) ---

class FontLoadError(Exception): pass
class FontDrawError(Exception): pass

def get_random_font(font_paths, exclude_list=None):
    """Selects a random font file path from the list, avoiding excluded ones."""
    available_fonts = list(set(font_paths) - set(exclude_list or []))
    if not available_fonts:
        try:
            # More robust fallback finding sans-serif
            prop = fm.FontProperties(family='sans-serif')
            fallback_path = fm.findfont(prop, fallback_to_default=True)
            if fallback_path:
                 print(f"Warning: No usable fonts found from list/system. Using fallback: {fallback_path}")
                 return fallback_path
            else:
                 # If even matplotlib fallback fails (unlikely but possible)
                 print("ERROR: No fonts found in specified dir, system, or fallback. Cannot proceed.")
                 return None
        except Exception as e:
            print(f"ERROR: Font fallback mechanism failed: {e}. Cannot proceed.")
            return None
    return random.choice(available_fonts)

# Fallback random text generator
def generate_random_words(num_words):
    """Generates a string of random 'words' using only FALLBACK_CHAR_SET."""
    words = []
    for _ in range(num_words):
        length = random.randint(3, 8)
        word = ''.join(random.choice(FALLBACK_CHAR_SET.replace(" ", "")) for i in range(length))
        words.append(word)
    return " ".join(words)

def generate_random_text_snippet(highlighted_text, min_lines, max_lines, frame_number=None):
    """Generates multiple lines of random text, ensuring MIN_LINES."""
    # Use frame number as seed modifier for more variety
    if frame_number is not None and RANDOM_SEED_PER_FRAME:
        random.seed(int(time.time() * 1000) + frame_number)
    
    # Ensure we generate at least min_lines
    num_lines = random.randint(max(1, min_lines), max(min_lines, max_lines))  # Ensure at least min_lines generated
    highlight_line_index = random.randint(0, num_lines - 1)
    lines = []
    min_words_around = 2
    max_words_around = 6
    for i in range(num_lines):
        if i == highlight_line_index:
            words_before = generate_random_words(random.randint(min_words_around, max_words_around))
            words_after = generate_random_words(random.randint(min_words_around, max_words_around))
            lines.append(f"{words_before} {highlighted_text} {words_after}")
        else:
            lines.append(generate_random_words(random.randint(max_words_around, max_words_around * 2)))

    # Double-check final line count (should always pass with the adjusted randint)
    if len(lines) < min_lines:
        print(f"Warning: Random generator created only {len(lines)} lines (min: {min_lines}). This shouldn't happen.")
        return None, -1  # Treat as failure if check fails unexpectedly

    return lines, highlight_line_index

# IMPROVED: Enhanced Mistral AI Text Generation Function with Variety
def generate_ai_text_snippet_with_variety(client, model, highlighted_text, min_lines, max_lines, frame_number=0):
    """Generates a unique text snippet using Mistral AI with enhanced variety techniques."""
    target_lines = random.randint(min_lines, max_lines)
    
    # Create variety in temperature for different creativity levels
    temperature = random.uniform(*TEMPERATURE_RANGE)
    
    # Generate different theme contexts for variety
    theme_contexts = [
        "epic fantasy battle scenes",
        "royal court intrigue and politics",
        "ancient legends and prophecies",
        "magical realms and mystical powers",
        "heroic quests and adventures",
        "dragon lore and mythology",
        "medieval kingdoms and castles",
        "powerful rulers and their domains",
        "mystical ceremonies and rituals",
        "legendary warriors and champions"
    ] if highlighted_text.lower() in ["mother of dragons", "dragon", "dragons"] else [
        "leadership and authority themes",
        "powerful figures in history",
        "inspirational moments and speeches",
        "dramatic storytelling scenes",
        "character development narratives",
        "emotional and impactful moments",
        "pivotal plot developments",
        "memorable quotes and dialogue",
        "transformative story arcs",
        "compelling character interactions"
    ]
    
    # Select a random theme context if variety is enabled
    theme_context = random.choice(theme_contexts) if THEME_VARIETY else "fantasy, power, dragons, leadership"
    
    # Add frame-specific elements for uniqueness
    uniqueness_elements = [
        f"Frame perspective #{frame_number + 1}",
        f"Sequence element {frame_number % 5 + 1}",
        f"Story moment #{(frame_number // 3) + 1}",
        f"Scene variation {frame_number % 7 + 1}"
    ]
    uniqueness_hint = random.choice(uniqueness_elements)
    
    # Enhanced prompt with variety techniques
    prompt = (
        f"Create a unique text block of exactly {target_lines} distinct lines (minimum {min_lines} lines required). "
        f"One line MUST contain the exact phrase: '{highlighted_text}'. "
        f"Context theme: {theme_context}. "
        f"Uniqueness element: {uniqueness_hint}. "
        f"Make each line narratively distinct and avoid repetitive phrasing. "
        f"Ensure the phrase '{highlighted_text}' fits naturally within its line context. "
        f"Vary sentence structure, length, and perspective across lines. "
        f"Format as plain text lines separated by newlines only, no extra formatting or explanations."
    )

    try:
        messages = [UserMessage(content=prompt)]
        chat_response = client.chat.complete(
            model=model, 
            messages=messages,
            temperature=temperature,  # Use variable temperature
            max_tokens=350,  # Slightly increased for more content
            random_seed=int(time.time() * 1000 + frame_number) if RANDOM_SEED_PER_FRAME else None
        )
        content = chat_response.choices[0].message.content.strip()

        # Basic cleanup: remove potential empty lines
        lines = [line.strip() for line in content.split('\n') if line.strip()]

        # --- CRITICAL CHECK: Ensure minimum lines ---
        if len(lines) < min_lines:
            print(
                f"Warning: AI returned only {len(lines)} valid lines (minimum requested: {min_lines}). Retrying generation.")
            return None, -1  # Indicate failure due to insufficient lines

        # Find the highlight line
        highlight_line_index = -1
        for i, line in enumerate(lines):
            if highlighted_text in line:
                highlight_line_index = i
                break

        if highlight_line_index == -1:
            print(f"Warning: AI response did not contain the exact phrase '{highlighted_text}'.")
            # Try to insert it into a random line as fallback
            if lines:
                insert_line_idx = random.randint(0, len(lines) - 1)
                original_line = lines[insert_line_idx]
                # Insert the phrase naturally into the line
                words = original_line.split()
                if len(words) > 2:
                    insert_pos = random.randint(1, len(words) - 1)
                    words.insert(insert_pos, highlighted_text)
                    lines[insert_line_idx] = ' '.join(words)
                    highlight_line_index = insert_line_idx
                    print(f"Inserted phrase into line {insert_line_idx + 1} as fallback.")
                else:
                    return None, -1
            else:
                return None, -1

        return lines, highlight_line_index
    except Exception as e:
        print(f"An unexpected error occurred during AI text generation: {e}")
        return None, -1  # Indicate failure

# Legacy function for backwards compatibility - now with variety
def generate_ai_text_snippet(client, model, highlighted_text, min_lines, max_lines, frame_number=0):
    """Legacy wrapper that calls the enhanced variety function."""
    return generate_ai_text_snippet_with_variety(client, model, highlighted_text, min_lines, max_lines, frame_number)

def create_radial_blur_mask(width, height, center_x, center_y, sharp_radius, fade_radius):
    """Creates a grayscale mask for radial blur (sharp center, fades out)."""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        (center_x - sharp_radius, center_y - sharp_radius,
         center_x + sharp_radius, center_y + sharp_radius),
        fill=255
    )
    # Gaussian blur the sharp circle mask for a smooth falloff
    # Ensure fade radius is larger than sharp radius
    blur_amount = max(0.1, (fade_radius - sharp_radius) / 3.5)  # Adjusted divisor for smoothness
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_amount))
    return mask


def create_text_image_frame(width, height, text_lines, highlight_line_index, highlighted_text,
                            font_path, font_size, text_color, bg_color, highlight_color,
                            blur_type, blur_radius, radial_sharp_radius_factor, vertical_spread_factor):
    """Creates a single frame image with centered highlight and multi-line text."""

    # --- Font Loading ---
    try:
        font = ImageFont.truetype(font_path, font_size)
        bold_font = font  # Start with regular as fallback
        # Simple bold variant check (can be improved)
        common_bold_suffixes = ["bd.ttf", "-Bold.ttf", "b.ttf", "_Bold.ttf", " Bold.ttf"]
        base_name, ext = os.path.splitext(font_path)
        for suffix in common_bold_suffixes:
            potential_bold_path = base_name.replace("Regular", "").replace("regular",
                                                                           "") + suffix  # Try removing 'Regular' too
            if os.path.exists(potential_bold_path):
                try:
                    bold_font = ImageFont.truetype(potential_bold_path, font_size)
                    # print(f"    Using bold variant: {os.path.basename(potential_bold_path)}") # Debug
                    break  # Use the first one found
                except IOError:
                    continue  # Try next suffix if loading fails
            # Check without removing Regular if first checks failed
            potential_bold_path = base_name + suffix
            if os.path.exists(potential_bold_path):
                try:
                    bold_font = ImageFont.truetype(potential_bold_path, font_size)
                    # print(f"    Using bold variant: {os.path.basename(potential_bold_path)}") # Debug
                    break
                except IOError:
                    continue

    except IOError as e:
        raise FontLoadError(f"Failed to load font: {font_path}") from e
    except Exception as e:  # Catch other potential font loading issues
        raise FontLoadError(f"Unexpected error loading font {font_path}: {e}") from e

    # --- Calculations ---
    try:
        # Line height using getmetrics()
        try:
             ascent, descent = font.getmetrics()
             metric_height = ascent + abs(descent)
             line_height = int(metric_height * vertical_spread_factor)
        except AttributeError:
             bbox_line_test = font.getbbox("Ay", anchor="lt")
             line_height = int((bbox_line_test[3] - bbox_line_test[1]) * vertical_spread_factor)
        if line_height <= font_size * 0.8:
            line_height = int(font_size * 1.2 * vertical_spread_factor)

        # BOLD font metrics for final highlight placement
        highlight_width_bold = bold_font.getlength(highlighted_text)
        highlight_bbox_h = bold_font.getbbox(highlighted_text, anchor="lt")
        highlight_height_bold = highlight_bbox_h[3] - highlight_bbox_h[1]
        if highlight_width_bold <= 0 or highlight_height_bold <= 0:
             highlight_height_bold = int(font_size * 1.1)
             if highlight_width_bold <=0: highlight_width_bold = len(highlighted_text) * font_size * 0.6

        # Target position for the TOP-LEFT of the final BOLD highlight text (CENTERED)
        highlight_target_x = (width - highlight_width_bold) / 2
        highlight_target_y = (height - highlight_height_bold) / 2

        # Block start Y calculated relative to the centered highlight's top
        block_start_y = highlight_target_y - (highlight_line_index * line_height)

        # Get Prefix and Suffix for background alignment
        highlight_line_full_text = text_lines[highlight_line_index]
        prefix_text = ""
        suffix_text = "" # Also get suffix now
        highlight_found_in_line = False
        try:
            start_index = highlight_line_full_text.index(highlighted_text)
            end_index = start_index + len(highlighted_text)
            prefix_text = highlight_line_full_text[:start_index]
            suffix_text = highlight_line_full_text[end_index:]
            highlight_found_in_line = True
        except ValueError: pass # Treat line normally if not found

        # Measure Prefix Width using REGULAR font (for background positioning)
        prefix_width_regular = font.getlength(prefix_text)
        # Calculate the required starting X for the background highlight line string
        # This is the coordinate used for drawing the *full string* in the background
        bg_highlight_line_start_x = highlight_target_x - prefix_width_regular

    except AttributeError: raise FontDrawError(f"Font lacks methods.")
    except Exception as e: raise FontDrawError(f"Measurement fail: {e}") from e

    # --- Base Image Drawing (Draw FULL lines, use offset for HL line) ---
    # Render onto img_base normally first
    img_base = Image.new('RGB', (width, height), color=bg_color)
    draw_base = ImageDraw.Draw(img_base)
    try:
        current_y = block_start_y
        for i, line in enumerate(text_lines):
            line_x = 0.0
            if i == highlight_line_index and highlight_found_in_line:
                line_x = bg_highlight_line_start_x
            else:
                line_width = font.getlength(line)
                line_x = (width - line_width) / 2
            draw_base.text((line_x, current_y), line, font=font, fill=text_color, anchor="lt")
            current_y += line_height
    except Exception as e: raise FontDrawError(f"Base draw fail: {e}") from e

    # --- Apply Blur (with padding for Gaussian to avoid edge clipping) ---
    img_blurred = None # Initialize
    padding_for_blur = int(blur_radius * 3) # Padding based on blur radius

    if blur_type == 'gaussian' and blur_radius > 0:
        try:
            # Create larger canvas
            padded_width = width + 2 * padding_for_blur
            padded_height = height + 2 * padding_for_blur
            img_padded = Image.new('RGB', (padded_width, padded_height), color=bg_color)
            # Paste original centered onto padded canvas
            img_padded.paste(img_base, (padding_for_blur, padding_for_blur))
            # Blur the padded image
            img_padded_blurred = img_padded.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            # Crop the center back to original size
            img_blurred = img_padded_blurred.crop((padding_for_blur, padding_for_blur,
                                                  padding_for_blur + width, padding_for_blur + height))
        except Exception as e:
            print(f"Error during padded Gaussian blur: {e}. Falling back to direct blur.")
            img_blurred = img_base.filter(ImageFilter.GaussianBlur(radius=blur_radius)) # Fallback


    elif blur_type == 'radial' and blur_radius > 0:
        # For radial, we need img_sharp. Let's try drawing it *in parts* for reliability
        # as the padded blur trick doesn't apply directly here.
        img_sharp = Image.new('RGB', (width, height), color=bg_color)
        draw_sharp = ImageDraw.Draw(img_sharp)
        try:
            current_y = block_start_y
            for i, line in enumerate(text_lines):
                if i == highlight_line_index and highlight_found_in_line:
                    # --- Draw Sharp Highlight Line in Parts ---
                    # Calculate positions relative to the *final* centered highlight target
                    prefix_x = highlight_target_x - prefix_width_regular
                    # Use REGULAR font for the sharp layer (it's just for the mask)
                    draw_sharp.text((prefix_x, current_y), prefix_text, font=font, fill=text_color, anchor="lt")
                    # Highlight part itself starts at highlight_target_x
                    highlight_width_regular = font.getlength(highlighted_text) # Width in regular font
                    draw_sharp.text((highlight_target_x, current_y), highlighted_text, font=font, fill=text_color, anchor="lt")
                    # Suffix starts after the regular highlight width
                    suffix_x = highlight_target_x + highlight_width_regular
                    draw_sharp.text((suffix_x, current_y), suffix_text, font=font, fill=text_color, anchor="lt")
                else:
                    # Draw non-highlight lines centered normally
                    line_width = font.getlength(line)
                    line_x = (width - line_width) / 2
                    draw_sharp.text((line_x, current_y), line, font=font, fill=text_color, anchor="lt")
                current_y += line_height
        except Exception as e:
             raise FontDrawError(f"Failed sharp text draw (parts): {e}") from e

        # Composite blurred base and sharp center
        # Base image (img_base) still uses the offset drawing method for full line
        img_fully_blurred = img_base.filter(ImageFilter.GaussianBlur(radius=blur_radius * 1.5))
        sharp_center_radius = min(width, height) * radial_sharp_radius_factor
        fade_radius = sharp_center_radius + max(width, height) * 0.15
        mask = create_radial_blur_mask(width, height, width / 2, height / 2, sharp_center_radius, fade_radius)
        img_blurred = Image.composite(img_sharp, img_fully_blurred, mask)

    else: # No blur
        img_blurred = img_base.copy()


    # --- Final Image: Draw ONLY Highlight Rectangle & Centered BOLD Text ---
    final_img = img_blurred # Start with the blurred/composited image
    draw_final = ImageDraw.Draw(final_img)
    try:
        # 1. Draw highlight rectangle (centered using bold metrics)
        padding = font_size * 0.10
        draw_final.rectangle(
            [
                (highlight_target_x - padding, highlight_target_y - padding),
                (highlight_target_x + highlight_width_bold + padding, highlight_target_y + highlight_height_bold + padding)
            ],
            fill=highlight_color
        )

        # 2. Draw ONLY the SHARP highlight text using BOLD font at the *perfectly centered* position
        draw_final.text(
            (highlight_target_x, highlight_target_y),
            highlighted_text,
            font=bold_font, # Use BOLD font
            fill=text_color,
            anchor="lt"
        )
        # *** No prefix/suffix drawing here ***

    except Exception as e:
         raise FontDrawError(f"Failed final highlight draw: {e}") from e

    return final_img


# --- IMPROVED Core Video Generation Logic ---
def generate_video(params):
    """Generates the video based on input parameters with per-frame content generation."""

    # Unpack parameters from the dictionary passed by the Flask route
    width = params['width']
    height = params['height']
    fps = params['fps']
    duration_seconds = params['duration']
    highlighted_text = params['highlighted_text']
    highlight_color = params['highlight_color']
    text_color = params['text_color']
    background_color = params['background_color']
    blur_type = params['blur_type']
    blur_radius = params['blur_radius']
    ai_enabled = params['ai_enabled']
    font_dir = app.config['FONT_DIR'] # Use font dir from Flask config

    # Hardcoded or derived settings from original script
    font_size_ratio = 0.05 # Could be made a parameter
    min_lines = 7
    max_lines = 10
    vertical_spread_factor = 1.5
    radial_sharp_radius_factor = 0.3
    mistral_model = "mistral-large-latest" # Could be param

    print(f"Starting video generation with params: {params}")
    print(f"Enhanced mode: Generating unique content per frame")

    mistral_client = None
    if ai_enabled and MISTRAL_AVAILABLE:
        try:
            mistral_client = Mistral(api_key=MISTRAL_API_KEY)
            print("Mistral AI client initialized for per-frame generation.")
        except Exception as e:
            print(f"Error initializing Mistral client: {e}. Disabling AI for this request.")
            ai_enabled = False # Disable AI if client fails
    elif ai_enabled and not MISTRAL_AVAILABLE:
        print("AI was requested but is not available (check API key/library). Using random text.")
        ai_enabled = False

    # --- Font Discovery ---
    font_paths = []
    if font_dir and os.path.isdir(font_dir):
        print(f"Looking for fonts in specified directory: {font_dir}")
        for filename in os.listdir(font_dir):
            if filename.lower().endswith((".ttf", ".otf")):
                font_paths.append(os.path.join(font_dir, filename))
    else:
        print("FONT_DIR not specified or invalid, searching system fonts...")
        try:
            # Limit search to common locations if possible, or search all
            font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
            # font_paths.extend(fm.findSystemFonts(fontpaths=None, fontext='otf'))
        except Exception as e:
            print(f"Error finding system fonts: {e}")

    if not font_paths:
        print("ERROR: No fonts found in font dir or system. Cannot proceed.")
        return None, "No fonts found. Please add fonts to the 'fonts' directory or install system fonts."

    print(f"Found {len(font_paths)} potential fonts.")

    # --- Calculate Other Parameters ---
    total_frames = int(fps * duration_seconds)
    # Calculate font size based on height dynamically
    font_size = int(height * font_size_ratio)
    print(f"\nVideo Settings: {width}x{height} @ {fps}fps, {duration_seconds}s ({total_frames} frames)")
    print(f"Text Settings: Highlight='{highlighted_text}', Size={font_size}px")
    print(f"Effect Settings: BlurType='{blur_type}', BlurRadius={blur_radius}, HighlightColor='{highlight_color}'")
    print(f"Content Generation: {'AI (per-frame)' if ai_enabled else 'Random (per-frame)'}")

    # --- Generate Frames with Unique Content Per Frame ---
    frames = []
    failed_fonts = set()
    generation_failures = 0
    max_generation_failures = total_frames // 4  # Allow up to 25% generation failures
    
    print("\nGenerating frames with unique content...")
    frame_num = 0
    while frame_num < total_frames:
        # print(f"  Frame {frame_num + 1}/{total_frames}") # Can be verbose

        # GENERATE FRESH CONTENT FOR EACH FRAME
        current_lines = None
        highlight_idx = -1
        
        # Try to generate content for this specific frame
        content_attempts = 0
        max_content_attempts = 3
        
        while content_attempts < max_content_attempts and (current_lines is None or highlight_idx == -1):
            content_attempts += 1
            
            if ai_enabled and mistral_client:
                # Generate fresh AI content for this frame
                current_lines, highlight_idx = generate_ai_text_snippet_with_variety(
                    mistral_client, mistral_model, highlighted_text, min_lines, max_lines, frame_num
                )
                if current_lines is None or highlight_idx == -1:
                    print(f"    AI generation failed for frame {frame_num + 1}, attempt {content_attempts}")
                    if content_attempts == max_content_attempts:
                        print(f"    Falling back to random text for frame {frame_num + 1}")
                        current_lines, highlight_idx = generate_random_text_snippet(
                            highlighted_text, min_lines, max_lines, frame_num
                        )
            else:
                # Generate fresh random content for this frame
                current_lines, highlight_idx = generate_random_text_snippet(
                    highlighted_text, min_lines, max_lines, frame_num
                )
        
        # Check if content generation completely failed
        if current_lines is None or highlight_idx == -1:
            generation_failures += 1
            print(f"    WARNING: Content generation failed for frame {frame_num + 1}")
            
            if generation_failures > max_generation_failures:
                print(f"ERROR: Too many content generation failures ({generation_failures}). Stopping.")
                break
            else:
                frame_num += 1
                continue  # Skip this frame and try the next one

        # Now generate the visual frame with the unique content
        font_retries = 0
        frame_generated = False
        while font_retries < MAX_FONT_RETRIES_PER_FRAME:
            current_font_path = get_random_font(font_paths, exclude_list=failed_fonts)
            if current_font_path is None:
                # This now returns None only if EVERYTHING fails, including fallback
                return None, "No usable fonts available after multiple attempts."

            try:
                img = create_text_image_frame(
                    width, height,
                    current_lines, highlight_idx, highlighted_text,
                    current_font_path, font_size,
                    text_color, background_color, highlight_color,
                    blur_type, blur_radius, radial_sharp_radius_factor,
                    vertical_spread_factor
                )

                frame_np = np.array(img)
                frames.append(frame_np)
                frame_generated = True
                # print(f"    Frame {frame_num + 1} generated with unique content") # Less verbose
                break # Success, move to next frame attempt

            except (FontLoadError, FontDrawError) as e:
                print(f"    Warning: Font '{os.path.basename(current_font_path)}' failed for frame {frame_num + 1}. ({e}). Retrying with another font.")
                failed_fonts.add(current_font_path)
                font_retries += 1
                # Check if we've run out of fonts to try for this frame
                if len(failed_fonts) >= len(font_paths):
                     print(f"    ERROR: All available fonts failed for frame {frame_num + 1}. Trying system fallback once more.")
                     # Try the matplotlib fallback directly if list exhausted
                     fallback_font = get_random_font([], exclude_list=failed_fonts) # Trigger fallback explicitly
                     if fallback_font and fallback_font not in failed_fonts:
                          failed_fonts.add(fallback_font) # Add it so we don't retry infinitely
                          font_retries = 0 # Reset retries for the fallback font
                          print(f"    Attempting frame {frame_num + 1} with fallback font: {fallback_font}")
                          continue # Re-enter the loop to try drawing with fallback
                     else:
                          print(f"    ERROR: Even fallback font failed or wasn't found. Skipping frame {frame_num + 1}.")
                          frame_generated = False # Mark as not generated
                          break # Break font retry loop for this frame

            except Exception as e:
                print(f"    ERROR: Unexpected error generating frame {frame_num + 1} with font {os.path.basename(current_font_path)}: {e}")
                traceback.print_exc() # Log full error
                failed_fonts.add(current_font_path)
                font_retries += 1

        if not frame_generated:
            print(f"ERROR: Failed to generate Frame {frame_num + 1} after {MAX_FONT_RETRIES_PER_FRAME} font attempts. Stopping video generation.")
            # Decide whether to stop entirely or just make a shorter video
            # For a web app, stopping might be better than returning a broken/short video.
            return None, f"Failed to generate frame {frame_num + 1}. Font issues likely. Check font compatibility."
            # break # Or use break to create a shorter video

        frame_num += 1
        # Add progress update for long renders
        if frame_num % max(1, total_frames // 10) == 0 or frame_num == total_frames: # Update every 10%
             print(f"  Progress: {frame_num}/{total_frames} frames generated...")


    # --- Create Video ---
    if not frames:
        print("ERROR: No frames were generated. Cannot create video.")
        return None, "No frames were generated, possibly due to persistent font errors."

    if len(frames) < total_frames:
        print(f"Warning: Only {len(frames)}/{total_frames} frames were generated due to errors. Video will be shorter.")

    # Generate unique filename
    unique_id = uuid.uuid4()
    output_filename = f"text_match_cut_unique_{unique_id}.mp4"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

    print(f"\nCompiling video to {output_path}...")
    try:
        # Ensure frames is a list of numpy arrays
        if not isinstance(frames[0], np.ndarray):
             frames = [np.array(f) for f in frames]

        clip = ImageSequenceClip(frames, fps=fps)

        # Write video file using recommended settings
        # logger='bar' might not work well in web server logs, use None or default
        # Specify audio=False if there's no audio track
        # threads can speed up encoding, preset affects quality/speed balance
        clip.write_videofile(output_path,
                             codec='libx264', # Good compatibility
                             preset='medium', # Balance speed/quality (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
                             fps=fps,
                             threads=max(1, os.cpu_count() // 2), # Use half CPU cores
                             logger=None, # Avoid progress bar in server logs
                             audio=False) # Explicitly no audio

        clip.close() # Release resources
        print(f"\nVideo saved successfully as '{output_filename}' with unique content per frame")
        
        # Report generation statistics
        if generation_failures > 0:
            print(f"Content generation failures: {generation_failures}/{total_frames} frames")

        # Optionally list failed fonts
        if failed_fonts:
            print("\nFonts that caused errors during generation:")
            for ff in sorted(list(failed_fonts)):
                print(f" - {os.path.basename(ff)}")

        return output_filename, None # Return filename on success, no error

    except Exception as e:
        print(f"\nError during video writing: {e}")
        traceback.print_exc()
        error_message = f"Error during video writing: {e}. Check server logs and FFmpeg installation/codec support (libx264)."
        # Clean up potentially partially written file
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass # Ignore cleanup error
        return None, error_message


# --- Flask Routes (unchanged) ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main form page."""
    # Pass mistral availability to the template
    return render_template('index.html', mistral_available=MISTRAL_AVAILABLE)

@app.route('/generate', methods=['POST'])
def generate():
    """Handles form submission, triggers video generation."""
    try:
        params = {
            'width': request.form.get('width', default=1024, type=int),
            'height': request.form.get('height', default=1024, type=int),
            'fps': request.form.get('fps', default=10, type=int),
            'duration': request.form.get('duration', default=5, type=int),
            'highlighted_text': request.form.get('highlighted_text', default="Missing Text"),
            'highlight_color': request.form.get('highlight_color', default='#FFFF00'),
            'text_color': request.form.get('text_color', default='#000000'),
            'background_color': request.form.get('background_color', default='#FFFFFF'),
            'blur_type': request.form.get('blur_type', default='gaussian'),
            'blur_radius': request.form.get('blur_radius', default=4.0, type=float),
            'ai_enabled': request.form.get('ai_enabled') == 'true' and MISTRAL_AVAILABLE, # Only enable if checkbox checked AND available
        }

        # Basic Input Validation (Example)
        if not params['highlighted_text']:
            flash('Highlighted text cannot be empty.', 'error')
            return redirect(url_for('index'))
        if not (1 <= params['fps'] <= 60):
            flash('FPS must be between 1 and 60.', 'error')
            return redirect(url_for('index'))
        if not (1 <= params['duration'] <= 60): # Limit duration
             flash('Duration must be between 1 and 60 seconds.', 'error')
             return redirect(url_for('index'))
        if not (256 <= params['width'] <= 4096) or not (256 <= params['height'] <= 4096):
             flash('Width and Height must be between 256 and 4096 pixels.', 'error')
             return redirect(url_for('index'))


        # --- Trigger the generation ---
        generated_filename, error = generate_video(params)

        if error:
            # Render index page again, displaying the error
            return render_template('index.html', error=error, mistral_available=MISTRAL_AVAILABLE)
        else:
            # Render index page again, providing the download link
            return render_template('index.html', filename=generated_filename, mistral_available=MISTRAL_AVAILABLE)

    except Exception as e:
        print(f"An unexpected error occurred in /generate route: {e}")
        traceback.print_exc()
        return render_template('index.html', error=f"An unexpected server error occurred: {e}", mistral_available=MISTRAL_AVAILABLE)


@app.route('/output/<filename>')
def download_file(filename):
    """Serves the generated video file for download."""
    try:
        # Security: Ensure filename is safe and only serves from the UPLOAD_FOLDER
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)
    except FileNotFoundError:
         flash('Error: File not found. It might have been deleted or generation failed.', 'error')
         return redirect(url_for('index'))
    except Exception as e:
         print(f"Error serving file {filename}: {e}")
         flash('An error occurred while trying to serve the file.', 'error')
         return redirect(url_for('index'))

# --- Main Execution ---
if __name__ == '__main__':
    print(f"Mistral AI Available: {MISTRAL_AVAILABLE}")
    print("Enhanced Video Generator - Unique content per frame")
    # Use host='0.0.0.0' to make accessible on your network (use with caution)
    # debug=True automatically reloads on code changes, but disable for production
    app.run(debug=True, host='127.0.0.1', port=5000)