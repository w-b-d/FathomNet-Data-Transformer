"""
AI-assisted field mapping using Claude API.

Sends a dataset sample + directory structure + user prompt to Claude,
and receives a structured mapping configuration in return.
"""

import json
from typing import Optional

from anthropic import Anthropic

from fathomnet_schema import REQUIRED_FIELDS, OPTIONAL_FIELDS, ALL_FIELDS
from .mapping_config import MappingConfig


SYSTEM_PROMPT = """\
You are a dataset format analyzer for FathomNet, an ocean image database.

Your job: given a sample of an unknown dataset, figure out how to convert it
to FathomNet's required CSV format.

FathomNet REQUIRED columns:
- concept: species/object name (scientific names preferred)
- image: image filename or URL
- x: bounding box top-left X coordinate in pixels
- y: bounding box top-left Y coordinate in pixels
- width: bounding box width in pixels
- height: bounding box height in pixels

FathomNet OPTIONAL columns:
- depth, altitude, latitude, longitude
- temperature, salinity, oxygen, pressure
- observer, timestamp, imagingtype
- occluded, truncated, userdefinedkey, altconcept, groupof

RESPOND WITH ONLY a JSON object (no markdown, no explanation) with this structure:
{
    "source_format": "string — detected format name",
    "confidence": 0.0-1.0,
    "field_map": {
        "concept": {"source": "field_name_or_extraction_method", "transform": "optional"},
        "image": {"source": "..."},
        "x": {"source": "..."},
        "y": {"source": "..."},
        "width": {"source": "..."},
        "height": {"source": "..."}
    },
    "coordinate_format": "xywh|xyxy|cxcywh|cxcywh_abs",
    "exclude_concepts": ["list", "of", "classes", "to", "skip"],
    "concept_aliases": {"abbreviation": "Full Scientific Name"},
    "notes": "any important observations or warnings",
    "questions": ["questions for the user if anything is ambiguous"],
    "conversion_steps": ["step 1 description", "step 2 description"]
}

For field_map.source, use only conversion sources this tool can execute:
- A field name if the data has named fields (e.g., "class_name", "bbox[0]")
- "folder_name" if the concept comes from directory structure
- "filename_regex:<pattern>" if data is encoded in filenames

Do not use computed:<expression> or constant:<value>. If a conversion requires
computed or constant values that are not already handled by one of the built-in
converters, ask a question or explain the limitation in notes.
"""


class AIMapper:
    """Uses Claude API to analyze a dataset sample and produce a mapping config."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def analyze(
        self,
        sample: dict,
        detection_result: dict,
        user_prompt: str = "",
        correction_history: Optional[list] = None,
    ) -> dict:
        """
        Send dataset sample to Claude for analysis.

        Args:
            sample: output from sample_dataset()
            detection_result: output from detect_format()
            user_prompt: optional user description of the dataset
            correction_history: list of previous attempts + corrections

        Returns:
            dict with mapping config and any questions for the user
        """
        # Build the user message
        message_parts = []

        # File samples
        if sample.get("file_samples"):
            message_parts.append("=== FILE SAMPLES ===")
            for fs in sample["file_samples"]:
                message_parts.append(f"\n--- {fs['path']} ---")
                message_parts.append(fs["content_preview"])

        # Directory structure
        if sample.get("directory_tree"):
            message_parts.append("\n=== DIRECTORY STRUCTURE ===")
            message_parts.append(sample["directory_tree"])

        # Image filenames
        if sample.get("image_samples"):
            message_parts.append("\n=== SAMPLE IMAGE FILENAMES ===")
            for img in sample["image_samples"]:
                message_parts.append(f"  {img}")

        # Stats
        message_parts.append(f"\n=== STATS ===")
        message_parts.append(f"Images: {sample['stats'].get('image_count', 'unknown')}")
        message_parts.append(
            f"Annotation files: {sample['stats'].get('annotation_file_count', 0)}"
        )
        message_parts.append(
            f"Auto-detected format: {sample['stats'].get('format_detected', 'unknown')}"
        )

        # Detection details
        if detection_result.get("details"):
            message_parts.append(f"Detection notes: {detection_result['details']}")

        # User prompt
        if user_prompt:
            message_parts.append(f"\n=== USER DESCRIPTION ===")
            message_parts.append(user_prompt)

        # Correction history
        if correction_history:
            message_parts.append(f"\n=== PREVIOUS ATTEMPTS ===")
            for i, attempt in enumerate(correction_history, 1):
                message_parts.append(f"\nAttempt {i}:")
                message_parts.append(f"  Mapping used: {json.dumps(attempt.get('mapping', {}), indent=2)}")
                message_parts.append(f"  User feedback: {attempt.get('correction', 'none')}")

        user_message = "\n".join(message_parts)

        # Call Claude
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
        except TypeError as e:
            if "api_key" in str(e) or "auth_token" in str(e):
                return {
                    "error": "no_api_key",
                    "message": (
                        "No Anthropic API key found. AI-Assisted mode requires an API key.\n"
                        "Set it with:  export ANTHROPIC_API_KEY=sk-ant-your-key\n"
                        "Or use --mode known (auto-detects format, no API key needed)."
                    ),
                }
            raise

        # Parse response
        response_text = response.content[0].text.strip()

        # Try to extract JSON from response
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            import re
            json_match = re.search(r"\{[\s\S]*\}", response_text)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {
                    "error": "Could not parse AI response",
                    "raw_response": response_text,
                }

        return result

    def result_to_config(self, ai_result: dict) -> MappingConfig:
        """Convert AI analysis result to a MappingConfig."""
        config = MappingConfig(
            source_format=ai_result.get("source_format", "unknown"),
            field_map=ai_result.get("field_map", {}),
            exclude_concepts=ai_result.get("exclude_concepts", []),
            concept_aliases=ai_result.get("concept_aliases", {}),
            coordinate_format=ai_result.get("coordinate_format", "xywh"),
        )
        if ai_result.get("notes"):
            config.extra["ai_notes"] = ai_result["notes"]
        if ai_result.get("questions"):
            config.extra["ai_questions"] = ai_result["questions"]
        if ai_result.get("conversion_steps"):
            config.extra["conversion_steps"] = ai_result["conversion_steps"]
        return config
