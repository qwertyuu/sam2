# SAM 2 Video Annotator

Interactive web application for manually annotating video segments with SAM 2 before processing.

## Problem Solved

The original notebook had an automatic loop that propagated points from the last frame of one video segment to the next. This caused **tracking drift** where the model would gradually lose accuracy (e.g., tracking only a shirt instead of the whole person, missing hands).

This tool allows you to:
- Manually place annotation points on the first frame of each segment
- Save annotations without immediate processing (two-phase workflow)
- Preview ALL mask results in a grid before batch processing
- Have full control over annotation quality
- Prevent drift by ensuring clean initialization for each segment

## Features

### Phase 1: Annotation
- 🎯 **Interactive Canvas**: Click to place positive/negative points
- 💾 **Save Annotations**: Save points without processing (stored in JSON)
- 👁️ **Live Preview**: See mask results before saving
- 📊 **Progress Tracking**: Monitor which segments are annotated
- ⏭️ **Auto-Navigation**: Automatically loads next segment after saving
- 🎨 **Visual Feedback**: See all placed points with color coding
- 🔄 **Auto-Load**: Saved annotations load automatically when revisiting segments

### Phase 2: Review & Batch Processing
- 🖼️ **Grid Preview**: View all annotated segments' masks at once
- ✏️ **Easy Editing**: Click any segment to go back and re-annotate
- ⚡ **Batch Processing**: Process all segments in one go
- 💾 **Persistent Storage**: Annotations saved to disk, survive restarts
- 🎬 **Automatic Export**: Exports frames with green screen background

## Installation

1. Make sure SAM 2 is already installed (from the main repository)

2. Install additional dependencies:
```bash
cd web_annotator
pip install -r requirements.txt
```

## Configuration

Edit the `CONFIG` dictionary in `app.py` to match your setup:

```python
CONFIG = {
    'base_video_dir': '/path/to/organized',      # Where video_part_XXX folders are
    'output_dir': '/path/to/output',              # Where to save masked frames
    'checkpoint': 'checkpoints/sam2.1_hiera_large.pt',
    'model_config': 'configs/sam2.1/sam2.1_hiera_l.yaml',
    'num_segments': 62,                           # Total number of segments
    'start_segment': 22,                          # Which segment to start from
    'frame_offset': 19800,                        # Starting frame number for exports
    'background_color': [0, 255, 0]               # RGB for green screen
}
```

## Usage

### Quick Start

1. Start the web server:
```bash
cd H:\GitHub\sam2
python web_annotator/app.py
```

2. Open your browser to: `http://localhost:5000`

### Workflow: Three-Phase Process

#### **Phase 1: Annotate All Segments** (Main Page)

1. **Add Points:**
   - **Left-click** on the image to add positive points (regions to include)
   - **Right-click** to add negative points (regions to exclude)

2. **Preview (Optional):**
   - Click **"Preview Mask"** to see the segmentation result
   - Adjust points if needed

3. **Save:**
   - Click **"💾 Save Annotation"**
   - App automatically loads next segment
   - Repeat for all segments (22-62)

4. **Status:**
   - "Annotated Segments" counter shows progress
   - "Current Status" shows if segment is saved
   - Green checkmark (✓ Saved) means annotation is stored

#### **Phase 2: Review All Masks** (Review Page)

1. Click **"Review All"** button from main page
2. View grid of all annotated segments with mask overlays
3. Click any segment to go back and edit if mask looks wrong
4. When satisfied with all masks, proceed to Phase 3

#### **Phase 3: Batch Process** (Review Page)

1. Click **"▶️ Process All Segments"**
2. Backend processes all segments sequentially
3. Wait for completion (several minutes for 40 segments)
4. All frames exported to output directory

### Additional Features

- **Navigation:**
  - Use segment number input to jump to any segment
  - Click **"Next Segment"** to skip forward
  - Click **"Reset"** to clear all points

- **Immediate Processing:**
  - Click **"Process Now"** to process current segment immediately (bypasses batch)

- **Persistent Storage:**
  - Annotations saved to `web_annotator/annotations.json`
  - Restart app and your work is preserved

## How It Works

### Workflow

1. **Initialize**: Loads SAM 2 model and first frame of current segment
2. **Annotate**: You manually place points on the person/object to track
3. **Preview**: Optionally preview the mask on the first frame
4. **Process**:
   - Runs SAM 2 propagation through all frames in the segment
   - Exports each frame with green screen background
   - Only the masked regions keep original pixels
5. **Repeat**: Automatically loads next segment for annotation

### Key Differences from Original Notebook

| Original | Web Annotator |
|----------|---------------|
| Automatic point sampling from last frame | Manual point placement per segment |
| Blind propagation across segments | Visual inspection before processing |
| Accumulating drift over time | Clean initialization each segment |
| Hardcoded in notebook cells | Interactive web interface |

## Tips

- **Start with 3-5 positive points** on key parts of the subject (head, torso, hands)
- **Use negative points** if the mask includes unwanted regions
- **Preview before processing** to catch issues early
- **Consistent annotation** helps maintain quality across segments
- **Positive points** (green) = "Include this region"
- **Negative points** (red) = "Exclude this region"

## API Endpoints

The Flask backend provides these REST endpoints:

- `GET /api/init` - Initialize app and get first segment
- `POST /api/preview_mask` - Generate mask preview for current points
- `POST /api/process_segment` - Process entire segment with annotations
- `POST /api/skip_segment` - Skip to next segment
- `POST /api/load_segment` - Load specific segment number

## Troubleshooting

### Model not loading
- Check that checkpoint paths in CONFIG are correct
- Ensure SAM 2 is properly installed
- Verify CUDA/GPU is available if using GPU

### Segments not found
- Check that `base_video_dir` points to the correct location
- Verify folders are named `video_part_001`, `video_part_002`, etc.

### Slow processing
- Processing a 300-frame segment can take 2-5 minutes depending on GPU
- Consider using a smaller number of frames per segment
- Ensure you're using GPU acceleration

### Preview not showing
- Check browser console for JavaScript errors
- Verify the image is loading correctly in the canvas
- Try with fewer points first (complex masks take longer)

## Output

Masked frames are exported as:
- Format: `frame_XXXXXX_masked.png`
- Location: Specified in `CONFIG['output_dir']`
- Background: Green screen (customizable in config)
- Only masked regions contain original pixels

You can then create a video from these frames:
```bash
ffmpeg -f image2 -r 30 -i 'frame_%06d_masked.png' -vcodec libx264 -crf 22 output.mp4
```

## System Requirements

- Python 3.8+
- PyTorch 2.0+
- 8GB+ RAM (16GB+ recommended)
- GPU with 6GB+ VRAM (for reasonable speed)
- Modern web browser (Chrome, Firefox, Safari, Edge)
