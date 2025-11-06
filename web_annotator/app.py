"""
SAM 2 Video Segment Annotation Web Tool
Interactive web interface for manually annotating video segments before processing
"""

import os
import sys
import json
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from flask import Flask, render_template, request, jsonify
import cv2

# Ensure we're running from the correct directory for Hydra config resolution
# Change to the SAM2 repository root directory
sam2_root = Path(__file__).parent.parent
os.chdir(sam2_root)
print(f"Working directory set to: {sam2_root}")

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = Flask(__name__,
            template_folder=str(sam2_root / 'web_annotator' / 'templates'),
            static_folder=str(sam2_root / 'web_annotator' / 'static'))

# Configuration
CONFIG = {
    'base_video_dir': '/mnt/d/Boria/#5/#1 - Antoine/organized',
    'output_dir': '/mnt/d/Boria/#5/#1 - Antoine/extracted',
    'checkpoint': 'checkpoints/sam2.1_hiera_large.pt',
    'model_config': 'configs/sam2.1/sam2.1_hiera_l.yaml',
    'num_segments': 61,  # Last segment that exists
    'start_segment': 1,
    'frame_offset': 19800,
    'background_color': [0, 255, 0],  # Green screen RGB
    'annotations_file': 'web_annotator/annotations.json',  # Store annotations
    'previews_dir': 'web_annotator/previews'  # Store preview images
}

# Global state
state = {
    'video_predictor': None,  # For full segment processing with temporal propagation
    'image_predictor': None,  # For fast single-frame mask generation
    'device': None,
    'current_segment': CONFIG['start_segment'],
    'frame_offset': CONFIG['frame_offset'],
    'annotations': {}  # In-memory cache: {segment_num: {'points': [[x,y], ...], 'labels': [1,0,...]}}
}


def load_annotations():
    """Load annotations from JSON file"""
    annotations_path = CONFIG['annotations_file']
    if os.path.exists(annotations_path):
        with open(annotations_path, 'r') as f:
            # Convert string keys back to integers
            data = json.load(f)
            state['annotations'] = {int(k): v for k, v in data.items()}
            print(f"Loaded {len(state['annotations'])} annotations from {annotations_path}")
    else:
        state['annotations'] = {}
        print("No existing annotations file found")


def save_annotations():
    """Save annotations to JSON file"""
    annotations_path = CONFIG['annotations_file']
    os.makedirs(os.path.dirname(annotations_path), exist_ok=True)
    with open(annotations_path, 'w') as f:
        json.dump(state['annotations'], f, indent=2)
    print(f"Saved {len(state['annotations'])} annotations to {annotations_path}")


def generate_and_save_preview(segment_num, points, labels):
    """Generate mask preview and save to disk (optimized with image predictor)"""
    try:
        segment_dir = get_segment_dir(segment_num)
        frame_names = get_frame_names(segment_dir)

        # Load only the first frame (FAST - no need to load entire video)
        first_frame_path = os.path.join(segment_dir, frame_names[0])
        original_image = np.array(Image.open(first_frame_path).convert("RGB"))

        # Use image predictor for fast single-frame mask generation
        state['image_predictor'].set_image(original_image)

        # Predict mask using image predictor
        masks, scores, _ = state['image_predictor'].predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False,
        )

        # Get the mask (first and only mask since multimask_output=False)
        # Convert to boolean for array indexing (predict returns float 0.0/1.0)
        mask = masks[0].astype(bool)

        # Resize for grid display (smaller thumbnails)
        max_size = (400, 300)
        img_pil = Image.fromarray(original_image)
        img_pil.thumbnail(max_size, Image.Resampling.LANCZOS)
        resized_image = np.array(img_pil)

        # Resize mask to match
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (resized_image.shape[1], resized_image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # Create overlay (red with transparency)
        overlay = np.zeros_like(resized_image)
        overlay[mask_resized] = [255, 0, 0]  # Red

        # Blend
        alpha = 0.5
        blended = cv2.addWeighted(resized_image, 1 - alpha, overlay, alpha, 0)

        # Save to disk
        preview_dir = CONFIG['previews_dir']
        os.makedirs(preview_dir, exist_ok=True)
        preview_path = os.path.join(preview_dir, f"segment_{segment_num:03d}.jpg")

        blended_img = Image.fromarray(blended)
        blended_img.save(preview_path, format="JPEG", quality=85)

        print(f"Saved preview for segment {segment_num} to {preview_path}")

        # Return coverage for info
        return float(np.sum(mask_resized) / mask_resized.size)

    except Exception as e:
        print(f"Error generating preview for segment {segment_num}: {e}")
        return 0.0


def initialize_sam2():
    """Initialize SAM 2 models (both video and image predictors)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Get the correct path to checkpoints
    # We're already in the sam2 root directory
    checkpoint_path = CONFIG['checkpoint']

    # Pass the config path relative to sam2 root (where we changed directory to)
    model_config = CONFIG['model_config']

    # Build video predictor for full segment processing with temporal propagation
    video_predictor = build_sam2_video_predictor(
        model_config,
        checkpoint_path,
        device=device
    )

    # Build base model for image predictor (fast single-frame masks)
    sam2_model = build_sam2(
        model_config,
        checkpoint_path,
        device=device
    )
    image_predictor = SAM2ImagePredictor(sam2_model)

    state['video_predictor'] = video_predictor
    state['image_predictor'] = image_predictor
    state['device'] = device

    print("Initialized both video predictor (for segments) and image predictor (for previews)")

    return device


def get_segment_dir(segment_num):
    """Get the directory path for a specific segment"""
    return os.path.join(CONFIG['base_video_dir'], f"video_part_{segment_num:03d}")


def get_frame_names(video_dir):
    """Get sorted list of frame filenames from a video directory"""
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names


def image_to_base64(image_path):
    """Convert image to base64 string for web display"""
    with Image.open(image_path) as img:
        # Resize if too large for web display
        max_size = (1920, 1080)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"


def export_frame_with_masks(frame_idx, masks_dict, frame_path, output_path, bg_color=(0, 255, 0)):
    """
    Export a frame with masks applied and a colored background
    """
    original_image = Image.open(frame_path).convert("RGB")
    image_array = np.array(original_image)

    output_image = np.full_like(image_array, bg_color, dtype=np.uint8)
    combined_mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=bool)

    for obj_id, mask in masks_dict.items():
        mask = np.squeeze(mask)
        if mask.shape != combined_mask.shape:
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (combined_mask.shape[1], combined_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            mask = mask_resized.astype(bool)
        combined_mask = combined_mask | mask

    output_image[combined_mask] = image_array[combined_mask]

    result_image = Image.fromarray(output_image)
    result_image.save(output_path)

    return output_path


@app.route('/')
def index():
    """Render main annotation interface"""
    return render_template('index.html')


@app.route('/api/init', methods=['GET'])
def api_init():
    """Initialize the application and return current state"""
    try:
        if state['video_predictor'] is None or state['image_predictor'] is None:
            device = initialize_sam2()
            load_annotations()  # Load saved annotations

        segment_num = state['current_segment']
        segment_dir = get_segment_dir(segment_num)

        if not os.path.exists(segment_dir):
            return jsonify({
                'error': f"Segment directory not found: {segment_dir}"
            }), 404

        frame_names = get_frame_names(segment_dir)
        if not frame_names:
            return jsonify({
                'error': f"No frames found in {segment_dir}"
            }), 404

        # Get first frame
        first_frame_path = os.path.join(segment_dir, frame_names[0])
        image_data = image_to_base64(first_frame_path)

        # Get original image dimensions
        with Image.open(first_frame_path) as img:
            width, height = img.size

        # Check if this segment has saved annotation
        saved_annotation = state['annotations'].get(segment_num)
        annotated_segments = list(state['annotations'].keys())

        return jsonify({
            'segment_num': segment_num,
            'total_segments': CONFIG['num_segments'],
            'num_frames': len(frame_names),
            'image': image_data,
            'image_width': width,
            'image_height': height,
            'device': str(state['device']),
            'saved_annotation': saved_annotation,
            'annotated_segments': annotated_segments,
            'num_annotated': len(annotated_segments)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preview_mask', methods=['POST'])
def api_preview_mask():
    """Generate a preview of the mask based on current points (optimized with image predictor)"""
    try:
        data = request.json
        points = np.array(data['points'], dtype=np.float32)  # [[x, y], ...]
        labels = np.array(data['labels'], dtype=np.int32)  # [1, 0, 1, ...]
        segment_num = data['segment_num']

        segment_dir = get_segment_dir(segment_num)
        frame_names = get_frame_names(segment_dir)

        # Load only the first frame (FAST - no need to load entire video)
        first_frame_path = os.path.join(segment_dir, frame_names[0])
        original_image = np.array(Image.open(first_frame_path).convert("RGB"))

        # Use image predictor for fast single-frame mask generation
        state['image_predictor'].set_image(original_image)

        # Points are already in (x, y) format from frontend
        # Predict mask using image predictor
        masks, scores, _ = state['image_predictor'].predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False,
        )

        # Get the mask (first and only mask since multimask_output=False)
        # Convert to boolean for array indexing (predict returns float 0.0/1.0)
        mask = masks[0].astype(bool)

        # Create two previews:
        # 1. Red overlay (what will be KEPT)
        overlay_kept = np.zeros_like(original_image)
        overlay_kept[mask] = [255, 0, 0]  # Red for kept regions
        blended_kept = cv2.addWeighted(original_image, 1 - 0.5, overlay_kept, 0.5, 0)

        # 2. Green screen result (what final output looks like)
        green_bg = np.full_like(original_image, [0, 255, 0], dtype=np.uint8)
        green_bg[mask] = original_image[mask]

        # Convert to base64
        blended_img = Image.fromarray(blended_kept)
        buffered = BytesIO()
        blended_img.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        preview_overlay = f"data:image/jpeg;base64,{img_str}"

        greenscreen_img = Image.fromarray(green_bg)
        buffered2 = BytesIO()
        greenscreen_img.save(buffered2, format="JPEG", quality=90)
        img_str2 = base64.b64encode(buffered2.getvalue()).decode()
        preview_greenscreen = f"data:image/jpeg;base64,{img_str2}"

        return jsonify({
            'preview': preview_overlay,
            'preview_greenscreen': preview_greenscreen,
            'mask_coverage': float(np.sum(mask) / mask.size)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process_segment', methods=['POST'])
def api_process_segment():
    """Process a video segment with the provided points (uses video predictor for temporal propagation)"""
    try:
        data = request.json
        points = np.array(data['points'], dtype=np.float32)
        labels = np.array(data['labels'], dtype=np.int32)
        segment_num = data['segment_num']

        segment_dir = get_segment_dir(segment_num)
        frame_names = get_frame_names(segment_dir)

        # Initialize inference state with VIDEO predictor for temporal propagation
        inference_state = state['video_predictor'].init_state(video_path=segment_dir)

        # Points are already in (x, y) format - no reversal needed
        points_sam = points.copy()

        # Add points
        _, out_obj_ids, out_mask_logits = state['video_predictor'].add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=points_sam,
            labels=labels,
        )

        # Propagate through video
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in state['video_predictor'].propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Export frames
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        background_color = tuple(CONFIG['background_color'])
        exported_count = 0

        for frame_idx in range(len(frame_names) - 1):
            if frame_idx in video_segments:
                input_frame_path = os.path.join(segment_dir, frame_names[frame_idx])
                offset_frame_id = state['frame_offset'] + frame_idx
                output_frame_name = f"frame_{offset_frame_id:06d}_masked.png"
                output_frame_path = os.path.join(CONFIG['output_dir'], output_frame_name)

                export_frame_with_masks(
                    frame_idx,
                    video_segments[frame_idx],
                    input_frame_path,
                    output_frame_path,
                    background_color
                )
                exported_count += 1

        # Update frame offset and current segment
        state['frame_offset'] += exported_count
        state['current_segment'] += 1

        # Check if there's a next segment
        has_next = state['current_segment'] <= CONFIG['num_segments']

        return jsonify({
            'success': True,
            'exported_frames': exported_count,
            'next_segment': state['current_segment'] if has_next else None,
            'has_next': has_next
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/skip_segment', methods=['POST'])
def api_skip_segment():
    """Skip current segment and move to next"""
    state['current_segment'] += 1
    has_next = state['current_segment'] <= CONFIG['num_segments']

    return jsonify({
        'next_segment': state['current_segment'] if has_next else None,
        'has_next': has_next
    })


@app.route('/api/load_segment', methods=['POST'])
def api_load_segment():
    """Load a specific segment number"""
    try:
        data = request.json
        segment_num = data['segment_num']

        if segment_num < 1 or segment_num > CONFIG['num_segments']:
            return jsonify({'error': 'Invalid segment number'}), 400

        state['current_segment'] = segment_num

        segment_dir = get_segment_dir(segment_num)
        frame_names = get_frame_names(segment_dir)
        first_frame_path = os.path.join(segment_dir, frame_names[0])
        image_data = image_to_base64(first_frame_path)

        with Image.open(first_frame_path) as img:
            width, height = img.size

        # Check if this segment has saved annotation
        saved_annotation = state['annotations'].get(segment_num)

        return jsonify({
            'segment_num': segment_num,
            'num_frames': len(frame_names),
            'image': image_data,
            'image_width': width,
            'image_height': height,
            'saved_annotation': saved_annotation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/save_annotation', methods=['POST'])
def api_save_annotation():
    """Save annotation for a segment without processing"""
    try:
        data = request.json
        segment_num = data['segment_num']
        points = data['points']  # [[x, y], ...]
        labels = data['labels']  # [1, 0, ...]

        # Store annotation
        state['annotations'][segment_num] = {
            'points': points,
            'labels': labels
        }

        # Persist to disk
        save_annotations()

        # NOTE: We do NOT generate preview here - too slow (45 seconds)
        # Previews will be generated on-demand when viewing the review page

        return jsonify({
            'success': True,
            'segment_num': segment_num,
            'num_annotated': len(state['annotations'])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/delete_annotation/<int:segment_num>', methods=['DELETE'])
def api_delete_annotation(segment_num):
    """Delete a saved annotation"""
    try:
        if segment_num in state['annotations']:
            del state['annotations'][segment_num]
            save_annotations()

        return jsonify({
            'success': True,
            'num_annotated': len(state['annotations'])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/annotations_summary', methods=['GET'])
def api_annotations_summary():
    """Get summary of all annotations"""
    try:
        summary = []
        for segment_num in sorted(state['annotations'].keys()):
            annotation = state['annotations'][segment_num]
            summary.append({
                'segment_num': segment_num,
                'num_points': len(annotation['points']),
                'num_positive': sum(1 for l in annotation['labels'] if l == 1),
                'num_negative': sum(1 for l in annotation['labels'] if l == 0)
            })

        return jsonify({
            'annotations': summary,
            'total': len(summary)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/review')
def review():
    """Render review page"""
    return render_template('review.html')


@app.route('/api/preview_all', methods=['GET'])
def api_preview_all():
    """Load saved mask previews for all annotated segments"""
    try:
        previews = []
        preview_dir = CONFIG['previews_dir']

        for segment_num in sorted(state['annotations'].keys()):
            annotation = state['annotations'][segment_num]

            # Check if preview image exists
            preview_path = os.path.join(preview_dir, f"segment_{segment_num:03d}.jpg")

            if os.path.exists(preview_path):
                # Load saved preview and convert to base64
                with Image.open(preview_path) as img:
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG", quality=85)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    preview_data = f"data:image/jpeg;base64,{img_str}"
            else:
                # If preview doesn't exist, generate it now (fallback)
                print(f"Warning: Preview not found for segment {segment_num}, generating now...")
                points = np.array(annotation['points'], dtype=np.float32)
                labels = np.array(annotation['labels'], dtype=np.int32)
                generate_and_save_preview(segment_num, points, labels)

                # Now load it
                with Image.open(preview_path) as img:
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG", quality=85)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    preview_data = f"data:image/jpeg;base64,{img_str}"

            previews.append({
                'segment_num': segment_num,
                'preview': preview_data,
                'num_points': len(annotation['points']),
            })

        return jsonify({
            'previews': previews,
            'total': len(previews)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process_all', methods=['POST'])
def api_process_all():
    """Process all annotated segments in batch (uses video predictor for temporal propagation)"""
    try:
        if not state['annotations']:
            return jsonify({'error': 'No annotations to process'}), 400

        segments_to_process = sorted(state['annotations'].keys())
        total_segments = len(segments_to_process)
        results = []

        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        background_color = tuple(CONFIG['background_color'])

        for idx, segment_num in enumerate(segments_to_process, 1):
            print(f"Processing segment {segment_num} ({idx}/{total_segments})...")

            annotation = state['annotations'][segment_num]
            points = np.array(annotation['points'], dtype=np.float32)
            labels = np.array(annotation['labels'], dtype=np.int32)

            segment_dir = get_segment_dir(segment_num)
            frame_names = get_frame_names(segment_dir)

            # Initialize inference state with VIDEO predictor for temporal propagation
            inference_state = state['video_predictor'].init_state(video_path=segment_dir)

            # Points are already in (x, y) format - no reversal needed
            points_sam = points.copy()

            # Add points
            _, out_obj_ids, out_mask_logits = state['video_predictor'].add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points_sam,
                labels=labels,
            )

            # Propagate through video
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in state['video_predictor'].propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            # Export frames
            exported_count = 0
            for frame_idx in range(len(frame_names) - 1):
                if frame_idx in video_segments:
                    input_frame_path = os.path.join(segment_dir, frame_names[frame_idx])
                    offset_frame_id = state['frame_offset'] + frame_idx
                    output_frame_name = f"frame_{offset_frame_id:06d}_masked.png"
                    output_frame_path = os.path.join(CONFIG['output_dir'], output_frame_name)

                    export_frame_with_masks(
                        frame_idx,
                        video_segments[frame_idx],
                        input_frame_path,
                        output_frame_path,
                        background_color
                    )
                    exported_count += 1

            state['frame_offset'] += exported_count

            results.append({
                'segment_num': segment_num,
                'exported_frames': exported_count
            })

            print(f"Segment {segment_num} complete: {exported_count} frames exported")

        return jsonify({
            'success': True,
            'processed_segments': total_segments,
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting SAM 2 Video Annotator...")
    print(f"Base video directory: {CONFIG['base_video_dir']}")
    print(f"Output directory: {CONFIG['output_dir']}")
    print(f"Starting from segment: {CONFIG['start_segment']}")

    # Disable debug mode reloader to avoid path issues
    # You can still see errors in the browser and terminal
    app.run(debug=False, host='0.0.0.0', port=5000)
