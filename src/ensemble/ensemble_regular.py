"""
Multi-Model Ensemble Bias Detection System with Collaborative Discussion

This system combines three LLMs (Qwen3, Llama3.1, GPT-OSS) for political bias detection:
1. Individual Analysis: Each model analyzes articles independently
2. Consensus Check: System checks if models agree on direction
3. Collaborative Discussion: When all 3 models disagree, they engage in structured debate

Architecture:
- Sequential processing for memory efficiency
- Collaborative discussion only when all 3 directions differ (Left/Center/Right)
- Complete raw I/O logging for transparency
- Clean separation of concerns for maintainability
"""

import json
import os
import sys
import asyncio
import logging
import numpy as np
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import get_config

# Import robust JSON extractor
from src.utils.json_extractor import RobustJSONExtractor, clean_json_string, validate_discussion_json

# Import model classes
from src.models.qwen3_labeler import QwenLabeler
from src.models.gptoss_labeler import GPTOSSLabeler
from src.models.mistral_labeler import MistralLabeler
from src.utils.ground_truth import get_ground_truth_labels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def score_to_direction(score: float) -> str:
    """Convert bias score to direction (Left/Center/Right)."""
    if score <= -1:
        return "Left"
    elif score >= 1:
        return "Right"
    else:
        return "Center"


def get_ground_truth_text(article_data: dict, dataset_type: str) -> tuple:
    """Extract ground truth as text label (left/center/right)."""
    true_bias, is_valid = get_ground_truth_labels(article_data, dataset_type)
    if not is_valid:
        return None, False
    
    # Convert numeric to text: 0=Left, 1=Center, 2=Right
    mapping = {0: "left", 1: "center", 2: "right"}
    ground_truth_text = mapping.get(true_bias)
    return ground_truth_text, True


# ==============================================================================
# MODEL AGENT CLASS
# ==============================================================================

class ModelAgent:
    """
    Represents an individual model's analysis state during collaborative discussion.
    
    Attributes:
        agent_id: Unique identifier (qwen/llama/gptoss)
        initial_analysis: Original analysis from the model
        current_score: Current bias score (-3 to +3)
        current_reason: Current reasoning
        current_direction: Current direction (Left/Center/Right)
        discussion_history: List of previous states during discussion
    """
    
    def __init__(self, agent_id: str, initial_analysis: Dict[str, Any]):
        """Initialize agent with initial analysis from individual processing."""
        self.agent_id = agent_id
        self.initial_analysis = initial_analysis.copy()
        self.current_score = initial_analysis.get('score', 0)
        self.current_reason = initial_analysis.get('reason', '')
        self.current_direction = initial_analysis.get('direction', 'Center')
        self.discussion_history = []
        self.conversation_history = []  # Track full conversation for context
        
    def update_analysis(self, new_score: float, new_reason: str):
        """Update agent's analysis based on discussion."""
        # Save current state to history
        self.discussion_history.append({
            'score': self.current_score,
            'reason': self.current_reason,
            'direction': self.current_direction,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update current state
        self.current_score = new_score
        self.current_reason = new_reason
        self.current_direction = score_to_direction(new_score)
        
    def get_analysis_summary(self) -> str:
        """Get formatted summary of current analysis."""
        return f"Score: {self.current_score}, Direction: {self.current_direction}\nReason: {self.current_reason}"


# ==============================================================================
# MAIN ENSEMBLE SYSTEM CLASS
# ==============================================================================

class EnsembleMultiModelDetector:
    """
    Multi-model ensemble system for political bias detection with collaborative discussion.
    
    Uses Qwen3-14B, Llama3.1-70B-AWQ, and GPT-OSS-20B models sequentially, then applies
    consensus analysis and collaborative discussion for disagreements.
    """
    
    def __init__(self, config: Dict = None, batch_size: int = 3, dataset_type: str = 'baly',
                 output_dir: str = None, batch_start: int = None, batch_end: int = None):
        """
        Initialize the ensemble system.
        
        Args:
            config: Configuration dictionary
            batch_size: Batch size for processing articles
            dataset_type: Type of dataset being processed ('baly', 'budak', or 'ad_fontes')
        """
        if config is None:
            config = get_config()
        self.config = config
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        self.batch_start = batch_start
        self.batch_end = batch_end
        
        # Create session directory for outputs
        if output_dir:
            # Use provided output directory (for batch processing)
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Create new session directory (for standalone runs)
            self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path("ensemble_outputs") / f"session_{self.session_timestamp}"
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.individual_models_dir = self.output_dir / "individual_models"
        self.individual_models_dir.mkdir(parents=True, exist_ok=True)
        
        self.discussion_dir = self.output_dir / "collaborative_discussions"
        self.discussion_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize labelers from vLLM config (cheap to construct - just an OpenAI client)
        vllm_cfg = config['vllm']
        api_key = vllm_cfg['api_key']
        regular_cfg = vllm_cfg['regular_ensemble']

        self.qwen_labeler = QwenLabeler(
            base_url=regular_cfg['qwen3']['base_url'],
            model_id=regular_cfg['qwen3']['model_id'],
            api_key=api_key,
            enable_thinking=regular_cfg['qwen3']['enable_thinking'],
        )
        self.gptoss_labeler = GPTOSSLabeler(
            base_url=regular_cfg['gptoss']['base_url'],
            model_id=regular_cfg['gptoss']['model_id'],
            api_key=api_key,
        )
        self.mistral_labeler = MistralLabeler(
            base_url=regular_cfg['mistral']['base_url'],
            model_id=regular_cfg['mistral']['model_id'],
            api_key=api_key,
        )
        
        # Discussion parameters
        self.max_discussion_rounds = 8
        self.convergence_threshold = 0.5
        
        # Statistics tracking
        self.stats = {
            'total_articles': 0,
            'consensus_unanimous': 0,
            'consensus_majority': 0,
            'discussion_triggered': 0,
            'discussion_converged': 0,
            'articles_skipped': 0,
            'model_errors': {'qwen': 0, 'gptoss': 0, 'mistral': 0}
        }
        
        logger.info(f"Initialized EnsembleMultiModelDetector")
        logger.info(f"Session output directory: {self.output_dir}")

    # ==========================================================================
    # MAIN PROCESSING METHOD
    # ==========================================================================
    
    async def process_articles(self, articles: List[tuple]) -> List[Dict]:
        """
        Main entry point: Process articles through the ensemble system.
        
        Args:
            articles: List of (article_data, filename) tuples
            
        Returns:
            List of final results with consensus/discussion outcomes
        """
        self.stats['total_articles'] = len(articles)
        logger.info(f"Processing {len(articles)} articles through ensemble system")
        
        # Phase 1: Individual Analysis
        logger.info("=" * 60)
        logger.info("PHASE 1: Individual Model Analysis")
        logger.info("=" * 60)
        
        qwen_results, gptoss_results, mistral_results = self._run_individual_analysis(articles)
        
        # Validate results
        if not self._validate_results(qwen_results, gptoss_results, mistral_results):
            raise ValueError("Model results validation failed")
        
        # Calculate initial consensus for ALL articles (before discussion)
        logger.info("=" * 60)
        logger.info("INITIAL ENSEMBLE EVALUATION (Pre-Discussion)")
        logger.info("=" * 60)
        
        # Build initial consensus results
        initial_results = []
        for i in range(len(qwen_results)):
            consensus = self._check_consensus(qwen_results[i], gptoss_results[i], mistral_results[i], update_stats=False)
            initial_result = {
                'article_id': i,
                'filename': qwen_results[i]['filename'],
                'final_score': consensus.get('final_score'),
                'final_direction': consensus.get('final_direction'),
                'consensus_type': consensus['type']
            }
            initial_results.append(initial_result)
        
        # Calculate and display pre-discussion metrics
        logger.info("\nEnsemble Performance (Averaging/Majority Voting Only):")
        pre_discussion_metrics = self._calculate_evaluation_metrics(initial_results, articles, display_header=False)
        
        # Store pre-discussion metrics
        self.pre_discussion_metrics = pre_discussion_metrics
        
        # Phase 2: Consensus and Discussion
        logger.info("=" * 60)
        logger.info("PHASE 2: Collaborative Discussion for Disagreements")
        logger.info("=" * 60)
        
        final_results = []
        
        for i in range(len(qwen_results)):
            qwen_res = qwen_results[i]
            gptoss_res = gptoss_results[i]
            mistral_res = mistral_results[i]
            
            # Check consensus (update_stats=True by default)
            consensus = self._check_consensus(qwen_res, gptoss_res, mistral_res)
            
            # Prepare final result
            final_result = {
                'article_id': i,
                'filename': qwen_res['filename'],
                'models_used': ['qwen', 'gptoss', 'mistral'],
                'individual_scores': {
                    'qwen': {'score': qwen_res['score'], 'direction': qwen_res['direction']},
                    'gptoss': {'score': gptoss_res['score'], 'direction': gptoss_res['direction']},
                    'mistral': {'score': mistral_res['score'], 'direction': mistral_res['direction']}
                },
                'consensus_type': consensus['type'],
                'final_score': consensus.get('final_score'),
                'final_direction': consensus.get('final_direction')
            }
            
            # Phase 3: Collaborative Discussion (only if all 3 directions differ)
            if consensus['needs_discussion']:
                logger.info(f"Article {i}: Triggering collaborative discussion (all 3 models differ)")
                # Show individual scores for clarity
                score_details = f"qwen:{qwen_res['score']:.0f}, gptoss:{gptoss_res['score']:.0f}, mistral:{mistral_res['score']:.0f}"
                logger.info(f"Pre-discussion: Individual scores=[{score_details}], Avg={consensus['final_score']:.2f}, Direction={consensus['final_direction']}")
                
                # Store pre-discussion values
                pre_discussion_score = consensus['final_score']
                pre_discussion_direction = consensus['final_direction']
                
                # Get original article content
                article_content = articles[i][0]['content'] if i < len(articles) else ""
                if not article_content:
                    logger.error(f"Missing article content for article {i}")
                    continue
                
                try:
                    # Run collaborative discussion with 30-minute timeout
                    discussion_result = await asyncio.wait_for(
                        self._run_collaborative_discussion(
                            article_content, qwen_res, gptoss_res, mistral_res, article_id=i
                        ),
                        timeout=1800  # 30 minute timeout per discussion
                    )
                    
                    # Add pre-discussion values for comparison
                    discussion_result['pre_discussion_score'] = pre_discussion_score
                    discussion_result['pre_discussion_direction'] = pre_discussion_direction
                    
                    # Calculate change metrics
                    score_change = abs(discussion_result.get('final_score', pre_discussion_score) - pre_discussion_score)
                    direction_changed = discussion_result.get('final_direction') != pre_discussion_direction
                    
                    discussion_result['score_change'] = score_change
                    discussion_result['direction_changed'] = direction_changed
                    
                    logger.info(f"Post-discussion: Score={discussion_result.get('final_score', pre_discussion_score):.2f}, "
                              f"Direction={discussion_result.get('final_direction', pre_discussion_direction)}, "
                              f"Change={score_change:.2f}, Direction Changed={direction_changed}")
                    
                    final_result.update(discussion_result)
                    self.stats['discussion_triggered'] += 1
                    if discussion_result.get('convergence_achieved'):
                        self.stats['discussion_converged'] += 1
                        
                except asyncio.TimeoutError:
                    logger.error(f"Article {i}: Discussion timeout after 30 minutes - SKIPPING ARTICLE")
                    self.stats['articles_skipped'] += 1
                    self.stats['discussion_triggered'] += 1
                    continue  # Skip to next article
                    
                except (ValueError, RuntimeError) as e:
                    logger.error(f"Article {i}: Critical discussion error - SKIPPING ARTICLE: {e}")
                    self.stats['articles_skipped'] += 1
                    self.stats['discussion_triggered'] += 1  # Count as triggered but failed
                    continue  # Skip to next article
                    
                except Exception as e:
                    logger.error(f"Article {i}: Unexpected error - SKIPPING ARTICLE: {e}")
                    self.stats['articles_skipped'] += 1
                    continue  # Skip to next article
            
            final_results.append(final_result)
            
            # Save article results
            self._save_article_results(i, qwen_res, gptoss_res, mistral_res, final_result)
        
        # Save session summary
        self._save_session_summary(final_results)
        
        # Calculate post-discussion metrics
        logger.info("=" * 60)
        logger.info("POST-DISCUSSION EVALUATION")
        logger.info("=" * 60)
        logger.info("\nEnsemble Performance (After Collaborative Discussion):")
        post_discussion_metrics = self._calculate_evaluation_metrics(final_results, articles, display_header=False)
        
        # Display metrics comparison
        self._display_metrics_comparison(pre_discussion_metrics, post_discussion_metrics)
        
        # Display discussion impact summary
        self._display_discussion_impact(final_results)
        
        # Calculate and display comprehensive evaluation
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE EVALUATION ANALYSIS")
        logger.info("=" * 80)
        self._comprehensive_evaluation(final_results, articles)
        
        logger.info(f"Ensemble processing completed: {len(final_results)} articles processed")
        return final_results

    # ==========================================================================
    # PHASE 1: INDIVIDUAL ANALYSIS METHODS
    # ==========================================================================
    
    def _run_individual_analysis(self, articles: List[tuple]) -> Tuple[List, List, List]:
        """
        Run all three models sequentially with memory management.
        
        Returns:
            Tuple of (qwen_results, gptoss_results, mistral_results)
        """
        # Process Qwen3
        logger.info("Processing with Qwen3-14B...")
        qwen_results = self._process_single_model('qwen', articles, self.qwen_labeler)
        self._save_individual_results('qwen', qwen_results, articles)

        # Process GPT-OSS
        logger.info("Processing with GPT-OSS-20B...")
        gptoss_results = self._process_single_model('gptoss', articles, self.gptoss_labeler)
        self._save_individual_results('gptoss', gptoss_results, articles)

        # Process Mistral
        logger.info("Processing with Mistral-Small-Instruct-2409...")
        mistral_results = self._process_single_model('mistral', articles, self.mistral_labeler)
        self._save_individual_results('mistral', mistral_results, articles)
        
        return qwen_results, gptoss_results, mistral_results
    
    def _process_single_model(self, model_name: str, articles: List[tuple], labeler) -> List[Dict]:
        """Process articles through a single model (vLLM - direct HTTP calls, no DataLoader)."""
        logger.info(f"Processing {len(articles)} articles with {model_name}")

        results = []

        for article_idx, (article_data, filename) in enumerate(articles):
            article_content = article_data.get('content', '')
            if not article_content or len(article_content.strip()) < 100:
                logger.warning(f"{model_name}: Skipping article {article_idx} ({filename}) - content too short")
                self.stats['model_errors'][model_name] += 1
                continue

            try:
                result = labeler.predict(article_content)
            except Exception as e:
                logger.error(f"{model_name}: Error processing article {article_idx} ({filename}): {e}")
                self.stats['model_errors'][model_name] += 1
                continue

            if result.get('error'):
                self.stats['model_errors'][model_name] += 1
                continue

            raw_response = result.get('raw_response', '')
            reason = result.get('reason', 'No reason provided')

            if not raw_response:
                logger.warning(f"{model_name} returned empty raw_response for article {article_idx}")

            if not reason or len(reason.strip()) < 10:
                logger.warning(f"{model_name} returned invalid reason for article {article_idx}: {reason}")

            article_result = {
                'article_id': article_idx,
                'filename': filename,
                'model': model_name,
                'score': result.get('lean', 0),
                'reason': reason,
                'direction': score_to_direction(result.get('lean', 0)),
                'raw_response': raw_response,
                'thinking': result.get('_thinking', None) if model_name == 'qwen' else None,
            }
            results.append(article_result)

        logger.info(f"{model_name} completed: {len(results)} successful, {self.stats['model_errors'][model_name]} errors")
        return results

    # ==========================================================================
    # PHASE 2: CONSENSUS ANALYSIS METHODS
    # ==========================================================================
    
    def _check_consensus(self, qwen_res: Dict, gptoss_res: Dict, mistral_res: Dict, update_stats: bool = True) -> Dict:
        """
        Check consensus between three models.
        
        Returns:
            Dictionary with consensus analysis and whether discussion is needed
        """
        directions = [qwen_res['direction'], gptoss_res['direction'], mistral_res['direction']]
        scores = [qwen_res['score'], gptoss_res['score'], mistral_res['score']]
        
        direction_counts = Counter(directions)
        unique_directions = len(set(directions))
        
        consensus = {
            'directions': directions,
            'scores': scores,
            'needs_discussion': False
        }
        
        if unique_directions == 1:
            # All three agree
            consensus['type'] = 'unanimous'
            consensus['final_score'] = np.mean(scores)
            consensus['final_direction'] = directions[0]
            if update_stats:
                self.stats['consensus_unanimous'] += 1
            
        elif unique_directions == 2:
            # Majority agreement (2 agree, 1 differs)
            most_common_direction, count = direction_counts.most_common(1)[0]
            same_direction_scores = [s for s, d in zip(scores, directions) if d == most_common_direction]
            consensus['type'] = 'majority'
            consensus['final_score'] = np.mean(same_direction_scores)
            consensus['final_direction'] = most_common_direction
            if update_stats:
                self.stats['consensus_majority'] += 1
            
        else:
            # All three differ - need collaborative discussion
            consensus['type'] = 'all_different'
            consensus['needs_discussion'] = True
            # Calculate initial average as baseline before discussion
            consensus['final_score'] = np.mean(scores)
            # Use score-based direction as initial consensus
            consensus['final_direction'] = score_to_direction(consensus['final_score'])
            consensus['pre_discussion_score'] = consensus['final_score']
            consensus['pre_discussion_direction'] = consensus['final_direction']
        
        return consensus

    # ==========================================================================
    # PHASE 3: COLLABORATIVE DISCUSSION METHODS
    # ==========================================================================
    
    async def _run_collaborative_discussion(self, article_content: str, 
                                           qwen_res: Dict, gptoss_res: Dict, mistral_res: Dict,
                                           article_id: int = 0) -> Dict:
        """
        Run two-stage collaborative discussion between all three models.
        
        Stage 1: Debate until majority consensus (2/3 agree)
        Stage 2: Representative of winning direction debates with minority
        
        Args:
            article_content: Complete article text
            qwen_res, gptoss_res, mistral_res: Initial analysis from each model
            article_id: Article ID for logging
            
        Returns:
            Discussion results with final consensus
            
        Raises:
            ValueError: If critical data is missing or invalid
            RuntimeError: If discussion fails to reach valid conclusion
        """
        try:
            # VALIDATION: Ensure all required fields exist (NO DEFAULTS)
            for model_name, res in [('qwen', qwen_res), ('gptoss', gptoss_res), ('mistral', mistral_res)]:
                if 'score' not in res or res['score'] is None:
                    raise ValueError(f"Article {article_id}: {model_name} has no valid score")
                if 'direction' not in res or res['direction'] is None:
                    raise ValueError(f"Article {article_id}: {model_name} has no valid direction")
            
            # Store initial positions - NO DEFAULTS
            initial_positions = {
                'qwen': {
                    'score': qwen_res['score'],
                    'direction': qwen_res['direction'],
                    'reason': qwen_res.get('reason', '')
                },
                'gptoss': {
                    'score': gptoss_res['score'],
                    'direction': gptoss_res['direction'],
                    'reason': gptoss_res.get('reason', '')
                },
                'mistral': {
                    'score': mistral_res['score'],
                    'direction': mistral_res['direction'],
                    'reason': mistral_res.get('reason', '')
                }
            }
            
            # VALIDATION: Ensure all 3 directions are different
            initial_directions = [pos['direction'] for pos in initial_positions.values()]
            if len(set(initial_directions)) != 3:
                raise ValueError(
                    f"Article {article_id}: Two-stage discussion requires 3 different initial directions, "
                    f"got {initial_directions}"
                )
            
            # Create agents for each model
            agents = {
                'qwen': ModelAgent('qwen', qwen_res),
                'gptoss': ModelAgent('gptoss', gptoss_res),
                'mistral': ModelAgent('mistral', mistral_res)
            }
            
            # Create discussion directory for this article
            article_discussion_dir = self.discussion_dir / f"article_{article_id:04d}"
            article_discussion_dir.mkdir(exist_ok=True)
            
            discussion_history = []
            
            # ========== STAGE 1: Debate until majority consensus ==========
            stage1_winner_direction = None
            stage1_representative_id = None
            stage1_score = None
            
            logger.info(f"Article {article_id}: Starting STAGE 1 discussion")
            logger.info(f"Initial directions: qwen:{initial_positions['qwen']['direction']}, "
                       f"gptoss:{initial_positions['gptoss']['direction']}, "
                       f"mistral:{initial_positions['mistral']['direction']}")
            
            for round_num in range(1, self.max_discussion_rounds + 1):
                logger.info(f"Article {article_id} - Stage 1 Round {round_num}")
                
                # Check for consensus BEFORE generating challenges
                current_directions = [agent.current_direction for agent in agents.values()]
                direction_counts = Counter(current_directions)
                
                # Check for full consensus (all 3 models agree)
                if len(set(current_directions)) == 1:
                    # All models agree - consensus reached!
                    stage1_winner_direction = current_directions[0]
                    
                    # Calculate average score of all models
                    all_scores = [agent.current_score for agent in agents.values()]
                    stage1_score = sum(all_scores) / len(all_scores)
                    
                    # Find model that initially had this direction (for reference)
                    try:
                        stage1_representative_id, _ = self._find_initial_direction_model(
                            stage1_winner_direction, initial_positions
                        )
                    except ValueError:
                        # All converged to a new direction - pick middle score as representative
                        sorted_agents = sorted(agents.items(), key=lambda x: x[1].current_score)
                        stage1_representative_id = sorted_agents[1][0]
                    
                    logger.info(f"STAGE 1 COMPLETE - FULL CONSENSUS after {round_num} rounds")
                    logger.info(f"All models agree on {stage1_winner_direction}")
                    logger.info(f"Average score: {stage1_score:.2f}, Representative: {stage1_representative_id}")
                    break
                
                # Check if majority reached (2/3 models agree) - end Stage 1
                if max(direction_counts.values()) >= 2:
                    majority_direction = direction_counts.most_common(1)[0][0]
                    logger.info(f"Round {round_num}: MAJORITY REACHED for {majority_direction} "
                               f"({direction_counts[majority_direction]}/3 models)")
                    logger.info("Stage 1 complete - proceeding to Stage 2")
                    break
                
                # Continue Stage 1 debate if no consensus or majority
                challenger_id, target_id = self._select_discussion_pair(agents)
                challenger = agents[challenger_id]
                target = agents[target_id]
                
                logger.info(f"Stage 1 Round {round_num}: {challenger_id} challenges {target_id}")
                
                # Generate challenge
                challenge_prompt, challenge_response = await self._generate_challenge(
                    challenger, target, article_content, article_id, round_num
                )
                
                # Save challenge I/O
                self._save_discussion_io(
                    article_discussion_dir, round_num, 'challenge',
                    challenger_id, target_id, challenge_prompt, challenge_response
                )
                
                # Parse challenge response to extract the actual challenge text using robust extractor
                challenge_fields = RobustJSONExtractor.extract_challenge_fields(challenge_response)
                
                # Get the challenge text
                challenge_text = challenge_fields.get('challenge', challenge_response)
                
                # Log extraction status
                if challenge_fields and 'challenge' in challenge_fields and challenge_fields['challenge'] != challenge_response:
                    logger.info(f"Successfully extracted structured challenge from {challenger_id}")
                    # Also log if we successfully extracted other fields
                    if 'understanding' in challenge_fields:
                        logger.debug(f"{challenger_id} understanding: {challenge_fields['understanding'][:100]}...")
                else:
                    logger.warning(f"Could not extract structured challenge from {challenger_id}, using full response")
                    # Log a snippet of what we're using as fallback
                    logger.debug(f"Using as challenge text: {challenge_text[:200]}...")
                
                # Check if challenger adjusted their score after considering target's perspective
                if 'adjusted_lean' in challenge_fields and challenge_fields['adjusted_lean'] is not None:
                    adjusted_lean = self._ensure_numeric_score(
                        challenge_fields['adjusted_lean'], 
                        challenger.current_score, 
                        "adjusted_lean"
                    )
                    if abs(adjusted_lean - challenger.current_score) > 0.1:
                        # Create clear adjustment message
                        old_direction = challenger.current_direction
                        old_score = challenger.current_score
                        new_direction = score_to_direction(adjusted_lean)
                        
                        adjustment_msg = (
                            f"**SCORE ADJUSTMENT: {challenger_id} revised from "
                            f"{old_direction}({old_score:+.1f}) to "
                            f"{new_direction}({adjusted_lean:+.1f}) "
                            f"after considering {target_id}'s analysis**"
                        )
                        
                        # Update the challenger's score
                        challenger.update_analysis(adjusted_lean, f"Adjusted after considering {target_id}'s perspective")
                        
                        # Add adjustment message to ALL agents' conversation history
                        for agent in agents.values():
                            agent.conversation_history.append(("SYSTEM", adjustment_msg))
                        
                        # Log the adjustment
                        logger.info(adjustment_msg)
                        
                        # Don't check convergence here in Stage 1 - will check at start of next round
                
                # Update conversation history with the challenge text (after handling adjustments)
                challenger.conversation_history.append((f"{challenger_id}->{target_id}", challenge_text))
                target.conversation_history.append((f"{challenger_id}->{target_id}", challenge_text))
                # The third agent also needs to see this
                for aid, agent in agents.items():
                    if aid not in [challenger_id, target_id]:
                        agent.conversation_history.append((f"{challenger_id}->{target_id}", challenge_text))
                
                # Generate response
                response_prompt, response_result = await self._generate_response(
                    target, challenge_text, challenger, article_content, article_id, round_num
                )
                
                # Save response I/O
                self._save_discussion_io(
                    article_discussion_dir, round_num, 'response',
                    target_id, challenger_id, response_prompt, response_result['raw_response']
                )
                
                # Update target's analysis if changed
                old_target_score = target.current_score  # Save for history tracking
                if abs(response_result['new_score'] - target.current_score) > 0.1:
                    # Create clear adjustment message
                    old_direction = target.current_direction
                    old_score = target.current_score
                    new_score = response_result['new_score']
                    new_direction = score_to_direction(new_score)
                    
                    adjustment_msg = (
                        f"**SCORE ADJUSTMENT: {target_id} revised from "
                        f"{old_direction}({old_score:+.1f}) to "
                        f"{new_direction}({new_score:+.1f}) "
                        f"after responding to {challenger_id}'s challenge**"
                    )
                    
                    # Update the target's score
                    target.update_analysis(new_score, response_result['new_reason'])
                    
                    # Add adjustment message to ALL agents' conversation history
                    for agent in agents.values():
                        agent.conversation_history.append(("SYSTEM", adjustment_msg))
                    
                    # Log the adjustment
                    logger.info(adjustment_msg)
                    
                    # Check if response shows perspective-taking
                    reason_lower = response_result['new_reason'].lower()
                    shows_understanding = any(phrase in reason_lower for phrase in [
                        'valid point', 'understand', 'see your perspective', 'acknowledge',
                        'you raise', 'considering', 'through their lens', 'interpretation'
                    ])
                    if shows_understanding:
                        logger.info(f"{target_id} shows perspective-taking in response")
                    
                    # Check for convergence immediately after agent updates position
                    current_directions_after_update = [agent.current_direction for agent in agents.values()]
                    if len(set(current_directions_after_update)) == 1:
                        # All models now agree - consensus reached!
                        stage1_winner_direction = current_directions_after_update[0]
                        all_scores = [agent.current_score for agent in agents.values()]
                        stage1_score = sum(all_scores) / len(all_scores)
                        
                        # Find representative
                        try:
                            stage1_representative_id, _ = self._find_initial_direction_model(
                                stage1_winner_direction, initial_positions
                            )
                        except ValueError:
                            sorted_agents = sorted(agents.items(), key=lambda x: x[1].current_score)
                            stage1_representative_id = sorted_agents[1][0]
                        
                        logger.info(f"STAGE 1 COMPLETE - CONVERGENCE AFTER UPDATE in round {round_num}")
                        logger.info(f"All models agree on {stage1_winner_direction}")
                        logger.info(f"Average score: {stage1_score:.2f}, Representative: {stage1_representative_id}")
                        
                        # Record this round in history before breaking
                        discussion_history.append({
                            'stage': 1,
                            'round': round_num,
                            'challenger': challenger_id,
                            'target': target_id,
                            'challenge': challenge_response,
                            'response_changed': True,
                            'converged': True,
                            'agent_states': {
                                aid: {'score': agent.current_score, 'direction': agent.current_direction}
                                for aid, agent in agents.items()
                            }
                        })
                        break  # Exit the round loop immediately
                    
                else:
                    logger.info(f"{target_id} maintains position at score: {target.current_score}")
                
                # Update conversation history with the response
                response_text = response_result.get('reason', response_result.get('new_reason', ''))
                for agent in agents.values():
                    agent.conversation_history.append((f"{target_id}->{challenger_id}", response_text))
                
                # Record Stage 1 round history (only if we didn't already record it during convergence)
                # Check if we just converged and already added to history
                if not (len(discussion_history) > 0 and 
                       discussion_history[-1].get('round') == round_num and 
                       discussion_history[-1].get('converged', False)):
                    discussion_history.append({
                        'stage': 1,
                        'round': round_num,
                        'challenger': challenger_id,
                        'target': target_id,
                        'challenge': challenge_response,
                        'response_changed': abs(response_result['new_score'] - old_target_score) > 0.1,
                        'agent_states': {
                            aid: {'score': agent.current_score, 'direction': agent.current_direction}
                            for aid, agent in agents.items()
                        }
                    })
            
            # After all rounds, determine Stage 1 outcome
            if stage1_winner_direction is None:
                # Check final state after all rounds
                final_directions = [agent.current_direction for agent in agents.values()]
                direction_counts = Counter(final_directions)
                
                if len(set(final_directions)) == 1:
                    # All agree - consensus reached!
                    stage1_winner_direction = final_directions[0]
                    all_scores = [agent.current_score for agent in agents.values()]
                    stage1_score = sum(all_scores) / len(all_scores)
                    
                    try:
                        stage1_representative_id, _ = self._find_initial_direction_model(
                            stage1_winner_direction, initial_positions
                        )
                    except ValueError:
                        sorted_agents = sorted(agents.items(), key=lambda x: x[1].current_score)
                        stage1_representative_id = sorted_agents[1][0]
                    
                    logger.info(f"Stage 1 reached CONSENSUS at round limit: All agree on {stage1_winner_direction}")
                    logger.info(f"Average score: {stage1_score:.2f}")
                    
                elif max(direction_counts.values()) >= 2:
                    # Majority exists - select representative for Stage 2
                    stage1_winner_direction = direction_counts.most_common(1)[0][0]
                    agreeing_models = [aid for aid, agent in agents.items() 
                                     if agent.current_direction == stage1_winner_direction]
                    
                    # Find model that initially had this direction
                    initial_model = None
                    for aid in agreeing_models:
                        if initial_positions[aid]['direction'] == stage1_winner_direction:
                            initial_model = aid
                            break
                    
                    if initial_model:
                        stage1_representative_id = initial_model
                    else:
                        # Pick model with strongest conviction from agreeing models
                        stage1_representative_id = max(agreeing_models, 
                                                      key=lambda x: abs(agents[x].current_score))
                    
                    stage1_score = agents[stage1_representative_id].current_score
                    
                    logger.info(f"Stage 1 ended with MAJORITY after {self.max_discussion_rounds} rounds")
                    logger.info(f"Majority direction: {stage1_winner_direction}, Representative: {stage1_representative_id}")
                    
                else:
                    # No majority - use conviction to determine representative
                    sorted_by_conviction = sorted(agents.items(), 
                                                 key=lambda x: abs(x[1].current_score), 
                                                 reverse=True)
                    stage1_representative_id = sorted_by_conviction[0][0]
                    stage1_winner_direction = sorted_by_conviction[0][1].current_direction
                    stage1_score = sorted_by_conviction[0][1].current_score
                    
                    logger.info(f"Stage 1 ended with NO MAJORITY after {self.max_discussion_rounds} rounds")
                    logger.info(f"Using highest conviction: {stage1_representative_id} with {stage1_winner_direction}")
            
            # ========== Check if Stage 2 is needed ==========
            # Count how many models agree on the winning direction
            final_directions = [agent.current_direction for agent in agents.values()]
            
            if len(set(final_directions)) == 1:
                # All models agree - no Stage 2 needed!
                final_score = stage1_score  # Use the average score from Stage 1
                final_direction = stage1_winner_direction
                
                logger.info(f"CONSENSUS REACHED - No Stage 2 needed")
                logger.info(f"All models agree on {final_direction} with average score {final_score:.2f}")
                
                # Save complete discussion summary
                self._save_discussion_summary(
                    article_discussion_dir, agents, discussion_history,
                    True, 'consensus_stage1'
                )
                
                return {
                    'discussion_method': 'collaborative_consensus',
                    'consensus_direction': final_direction,
                    'final_score': final_score,
                    'final_direction': final_direction,
                    'rounds_to_consensus': len(discussion_history),
                    'stage2_needed': False
                }
            
            # ========== STAGE 2: Winner vs Minority ==========
            # Stage 2 is triggered when we have majority (2 vs 1) after Stage 1
            logger.info(f"Stage 2 TRIGGERED - No consensus reached in Stage 1")
            logger.info(f"Current directions: {final_directions}")
            
            # Find the minority model(s) (those that didn't converge to majority)
            minority_models = [aid for aid, agent in agents.items() 
                              if agent.current_direction != stage1_winner_direction]
            
            if len(minority_models) == 0:
                # This shouldn't happen as we already checked for consensus
                raise RuntimeError(
                    f"Article {article_id}: Logic error - all models agree but consensus check failed"
                )
            elif len(minority_models) == 1:
                # Standard case: 2 vs 1 majority
                logger.info(f"Stage 2: Majority (2 vs 1) - {stage1_representative_id} vs {minority_models[0]}")
            elif len(minority_models) == 2:
                # All three models have different directions or no clear majority
                # In this case, stage1_winner was determined by conviction
                # Select the strongest dissenter as minority representative
                minority_id = max(minority_models, key=lambda x: abs(agents[x].current_score))
                minority_models = [minority_id]
                logger.info(f"Stage 2: No majority - using highest conviction representatives")
                logger.info(f"{stage1_representative_id} vs {minority_id}")
            else:
                raise RuntimeError(
                    f"Article {article_id}: Unexpected state - {len(minority_models)} minority models"
                )
            
            minority_id = minority_models[0]
            minority_agent = agents[minority_id]
            
            logger.info(f"Article {article_id}: Starting STAGE 2")
            logger.info(f"{stage1_representative_id}({stage1_winner_direction}, score:{stage1_score:.2f}) vs "
                       f"{minority_id}({minority_agent.current_direction}, score:{minority_agent.current_score:.2f})")
            
            # Run Stage 2 debate
            stage2_winner = await self._run_stage2_debate(
                agents, stage1_representative_id, minority_id,
                article_content, article_id, article_discussion_dir
            )
            
            # Determine final result based on Stage 2 winner
            # Winner takes all - all agents adopt the winner's position
            if stage2_winner == stage1_representative_id:
                final_score = agents[stage1_representative_id].current_score
                final_direction = agents[stage1_representative_id].current_direction
                logger.info(f"FINAL RESULT: Stage 1 winner {stage1_representative_id} won Stage 2")
                # Minority should adopt representative's position (may already be updated)
                if agents[minority_id].current_direction != final_direction:
                    agents[minority_id].update_analysis(final_score, f"Adopted {stage1_representative_id}'s position after Stage 2")
            else:
                # Minority won - ALL agents adopt minority's position
                final_score = agents[minority_id].current_score
                final_direction = agents[minority_id].current_direction
                logger.info(f"FINAL RESULT: Minority {minority_id} won Stage 2")
                
                # Update all other agents to match minority's winning position
                for aid, agent in agents.items():
                    if aid != minority_id and agent.current_direction != final_direction:
                        agent.update_analysis(final_score, f"Adopted {minority_id}'s position after Stage 2 victory")
                        logger.info(f"Updated {aid} to match Stage 2 winner {minority_id}: {final_direction} ({final_score:.2f})")
            
            logger.info(f"Final direction: {final_direction}, Final score: {final_score:.2f}")
            
            # Save complete discussion summary
            self._save_discussion_summary(
                article_discussion_dir, agents, discussion_history,
                True, 'two_stage_complete'
            )
            
            return {
                'discussion_method': 'collaborative_two_stage',
                'stage1_winner': stage1_winner_direction,
                'stage1_representative': stage1_representative_id,
                'stage1_score': stage1_score,
                'stage2_winner': stage2_winner,
                'final_score': final_score,
                'final_direction': final_direction,
                'discussion_rounds': len(discussion_history) + 1,  # +1 for Stage 2
                'convergence_achieved': True,
                'convergence_type': 'two_stage_complete',
                'final_agent_states': {
                    aid: {'score': agent.current_score, 'direction': agent.current_direction}
                    for aid, agent in agents.items()
                }
            }
            
        except (ValueError, RuntimeError) as e:
            # These are fundamental errors - article must be skipped
            logger.error(f"CRITICAL ERROR - Article {article_id} must be skipped: {e}")
            raise  # Re-raise to be handled by caller
            
        except Exception as e:
            # Unexpected errors
            logger.error(f"UNEXPECTED ERROR in article {article_id} discussion: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise  # Re-raise to be handled by caller
    
    async def _generate_challenge(self, challenger: ModelAgent, target: ModelAgent, 
                                 article_content: str, article_id: int, round_num: int) -> Tuple[str, str]:
        """
        Generate a challenge from one model to another.
        
        Returns:
            Tuple of (prompt, response)
        """
        # Build conversation history
        conversation_text = ""
        if challenger.conversation_history:
            conversation_text = "\n\nGROUP DISCUSSION HISTORY:\n"
            for i, (speaker, text) in enumerate(challenger.conversation_history):
                round_label = f"Round {i//2 + 1}"
                conversation_text += f"{round_label} - {speaker}: {text}\n"

        # Get appropriate labeler
        if challenger.agent_id == 'qwen':
            labeler = self.qwen_labeler
        elif challenger.agent_id == 'gptoss':
            labeler = self.gptoss_labeler
        else:
            labeler = self.mistral_labeler
        
        # Use the new discussion-specific method (returns prompt and response)
        prompt, response = labeler.generate_discussion_challenge(
            article_content=article_content,
            conversation_history=conversation_text,
            own_analysis={
                'score': challenger.current_score,
                'direction': challenger.current_direction,
                'reason': challenger.current_reason
            },
            target_analysis={
                'score': target.current_score,
                'direction': target.current_direction,
                'reason': target.current_reason
            }
        )
        
        # Note: adjusted_lean extraction and handling is now done in the main discussion loop
        # to enable immediate convergence checking
        
        # Check for empty response
        if not response or len(response.strip()) < 10:
            logger.error(f"Empty or very short challenge from {challenger.agent_id}")
            response = f"[{challenger.agent_id} failed to provide a valid challenge]"
        
        # Note: Conversation history is now updated in the main discussion loop
        # after extracting the challenge text and handling score adjustments

        return prompt, response

    async def _generate_response(self, responder: ModelAgent, challenge: str, challenger: ModelAgent,
                                article_content: str, article_id: int, round_num: int) -> Tuple[str, Dict]:
        """
        Generate a response to a challenge.
        
        Returns:
            Tuple of (prompt, response_dict with new_score, new_reason, raw_response)
        """
        # Build conversation history
        conversation_text = ""
        if responder.conversation_history:
            conversation_text = "\n\nGROUP DISCUSSION HISTORY:\n"
            for i, (speaker, text) in enumerate(responder.conversation_history[:-1]):  # Exclude last challenge (shown separately)
                round_label = f"Round {i//2 + 1}"
                conversation_text += f"{round_label} - {speaker}: {text}\n"

        # Get appropriate labeler
        if responder.agent_id == 'qwen':
            labeler = self.qwen_labeler
        elif responder.agent_id == 'gptoss':
            labeler = self.gptoss_labeler
        else:
            labeler = self.mistral_labeler
        
        # Use the new discussion-specific method (returns prompt and result)
        prompt, result = labeler.generate_discussion_response(
            article_content=article_content,
            conversation_history=conversation_text,
            challenge=challenge,
            own_analysis={
                'score': responder.current_score,
                'direction': responder.current_direction,
                'reason': responder.current_reason
            },
            challenger_analysis={
                'score': challenger.current_score,
                'direction': challenger.current_direction,
                'reason': challenger.current_reason
            }
        )
        
        # Extract values from result
        new_score = result.get('lean', responder.current_score)
        new_score = self._ensure_numeric_score(new_score, responder.current_score, "new_score")
        new_reason = result.get('reason', responder.current_reason)
        raw_response = result.get('raw_response', '')
        
        # Check for empty response
        if not new_reason or len(new_reason.strip()) < 10:
            logger.error(f"Empty or very short response from {responder.agent_id}")
            new_reason = f"[{responder.agent_id} failed to provide a valid response]"
        
        # Ensure score is in valid range
        new_score = max(-3, min(3, new_score))
        
        # Log warning if reason is very long but don't truncate
        if len(new_reason) > 5000:
            logger.warning(f"Long response reason from {responder.agent_id}: {len(new_reason)} characters")
        
        # Update conversation history for both agents
        responder.conversation_history.append((f"{responder.agent_id}->{challenger.agent_id}", new_reason))
        challenger.conversation_history.append((f"{responder.agent_id}->{challenger.agent_id}", new_reason))

        return prompt, {
            'new_score': new_score,
            'new_reason': new_reason,
            'raw_response': raw_response
        }

    async def _run_stage2_debate(self, agents: Dict[str, ModelAgent],
                                representative_id: str, minority_id: str,
                                article_content: str, article_id: int,
                                discussion_dir: Path) -> str:
        """
        Run Stage 2 debate between Stage 1 winner and minority.
        Determine winner based on direction convergence.
        
        Args:
            agents: Dictionary of all model agents
            representative_id: ID of Stage 1 representative model
            minority_id: ID of minority model
            article_content: Article text
            article_id: Article ID for logging
            discussion_dir: Directory to save discussion I/O
            
        Returns:
            Winner model ID (either representative_id or minority_id)
        """
        representative = agents[representative_id]
        minority = agents[minority_id]
        
        # Store pre-debate positions (current state after Stage 1)
        rep_direction_before = representative.current_direction
        minority_direction_before = minority.current_direction
        
        # Store INITIAL positions (from before any discussion)
        rep_initial_direction = representative.initial_analysis['direction']
        minority_initial_direction = minority.initial_analysis['direction']
        
        logger.info(f"Stage 2 START - Initial directions: {representative_id}:{rep_initial_direction}, {minority_id}:{minority_initial_direction}")
        logger.info(f"Stage 2 START - Current directions: {representative_id}:{rep_direction_before}, {minority_id}:{minority_direction_before}")
        
        # Generate challenge from representative
        challenge_prompt, challenge_response = await self._generate_challenge(
            representative, minority, article_content, article_id, round_num=999  # Special Stage 2 indicator
        )
        
        self._save_discussion_io(
            discussion_dir, 999, 'stage2_challenge',
            representative_id, minority_id, challenge_prompt, challenge_response
        )
        
        # Extract any score adjustment from challenge
        challenge_fields = RobustJSONExtractor.extract_challenge_fields(challenge_response)
        if 'adjusted_lean' in challenge_fields and challenge_fields['adjusted_lean'] is not None:
            adjusted_score = self._ensure_numeric_score(
                challenge_fields['adjusted_lean'],
                representative.current_score,
                "adjusted_lean"
            )
            if abs(adjusted_score - representative.current_score) > 0.1:
                representative.update_analysis(adjusted_score, "Adjusted during Stage 2 challenge")
        
        # Generate response from minority
        challenge_text = challenge_fields.get('challenge', challenge_response)
        response_prompt, response_result = await self._generate_response(
            minority, challenge_text, representative, article_content, article_id, round_num=999
        )
        
        self._save_discussion_io(
            discussion_dir, 999, 'stage2_response',
            minority_id, representative_id, response_prompt, response_result['raw_response']
        )
        
        # Update minority if changed
        minority_new_score = response_result.get('new_score', minority.current_score)
        if abs(minority_new_score - minority.current_score) > 0.1:
            minority.update_analysis(minority_new_score, response_result.get('new_reason', ''))
        
        # DETERMINE WINNER BASED ON DIRECTION CONVERGENCE
        rep_direction_after = representative.current_direction
        minority_direction_after = minority.current_direction
        
        logger.info(f"Stage 2 RESULT: {representative_id}:{rep_direction_before}->{rep_direction_after}, "
                   f"{minority_id}:{minority_direction_before}->{minority_direction_after}")
        
        # Case 1: Minority changed to representative's INITIAL direction
        if minority_direction_after == rep_initial_direction and minority_direction_before != rep_initial_direction:
            logger.info(f"Stage 2: {representative_id} wins - {minority_id} converged to {representative_id}'s initial direction {rep_initial_direction}")
            return representative_id
        
        # Case 2: Representative changed to minority's INITIAL direction
        if rep_direction_after == minority_initial_direction and rep_direction_before != minority_initial_direction:
            logger.info(f"Stage 2: {minority_id} wins - {representative_id} converged to {minority_id}'s initial direction {minority_initial_direction}")
            return minority_id
        
        # Case 3: Both converged to same direction
        if rep_direction_after == minority_direction_after:
            # CORRECTED LOGIC: Winner is whoever's INITIAL direction they converged to
            converged_direction = rep_direction_after
            
            if converged_direction == rep_initial_direction:
                logger.info(f"Stage 2: {representative_id} wins - both converged to {representative_id}'s initial direction {rep_initial_direction}")
                return representative_id
            elif converged_direction == minority_initial_direction:
                logger.info(f"Stage 2: {minority_id} wins - both converged to {minority_id}'s initial direction {minority_initial_direction}")
                return minority_id
            else:
                # Edge case: converged to a third direction (shouldn't happen in Stage 2)
                logger.warning(f"Stage 2: Both converged to {converged_direction}, which is neither's initial direction")
                # Use conviction as tiebreaker
                rep_conviction = abs(representative.current_score)
                minority_conviction = abs(minority.current_score)
                if rep_conviction >= minority_conviction:
                    logger.info(f"Stage 2: {representative_id} wins by conviction ({rep_conviction:.2f} >= {minority_conviction:.2f})")
                    return representative_id
                else:
                    logger.info(f"Stage 2: {minority_id} wins by conviction ({minority_conviction:.2f} > {rep_conviction:.2f})")
                    return minority_id
        
        # Case 4: No convergence - both maintain different directions
        # Use conviction (absolute score magnitude) as tiebreaker
        rep_conviction = abs(representative.current_score)
        minority_conviction = abs(minority.current_score)
        
        if rep_conviction > minority_conviction:
            logger.info(f"Stage 2: {representative_id} wins - no convergence, stronger conviction ({rep_conviction:.2f} > {minority_conviction:.2f})")
            return representative_id
        elif minority_conviction > rep_conviction:
            logger.info(f"Stage 2: {minority_id} wins - no convergence, stronger conviction ({minority_conviction:.2f} > {rep_conviction:.2f})")
            return minority_id
        else:
            # Rare tie - Stage 1 winner takes precedence
            logger.info(f"Stage 2: {representative_id} wins - conviction tie, Stage 1 winner takes precedence")
            return representative_id
    
    def _ensure_numeric_score(self, value, default_score: float, field_name: str = "score") -> float:
        """
        Ensure a score value is numeric, with error logging.
        
        Args:
            value: The value to convert
            default_score: Default score to use if conversion fails
            field_name: Name of the field for logging
            
        Returns:
            Numeric score value
        """
        if value is None:
            return default_score
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                numeric_value = float(value)
                logger.warning(f"Converted {field_name} from string '{value}' to float {numeric_value}")
                return numeric_value
            except ValueError:
                logger.error(f"Failed to convert {field_name} '{value}' to numeric, using default {default_score}")
                return default_score
        logger.error(f"Unexpected type for {field_name}: {type(value)}, using default {default_score}")
        return default_score
    
    def _find_initial_direction_model(self, direction: str, initial_positions: Dict) -> tuple:
        """
        Find which model initially had the given direction.
        
        Args:
            direction: The direction to search for (Left/Center/Right)
            initial_positions: Dict of initial model positions
            
        Returns:
            (model_id, initial_score) tuple
            
        Raises:
            ValueError: If no model initially had this direction or if score is missing
        """
        for model_id, pos in initial_positions.items():
            if pos.get('direction') == direction:
                score = pos.get('score')
                if score is None:
                    raise ValueError(f"Model {model_id} has no score in initial positions")
                return model_id, score
        
        # This should never happen in valid two-stage discussion
        raise ValueError(
            f"No model initially had direction '{direction}'. "
            f"This violates two-stage discussion logic where Stage 1 winner "
            f"must be one of the initial directions."
        )
    
    def _check_convergence(self, agents: Dict[str, ModelAgent]) -> Tuple[bool, str]:
        """Check if agents have converged to consensus."""
        directions = [agent.current_direction for agent in agents.values()]
        scores = [agent.current_score for agent in agents.values()]
        
        # Check unanimous agreement
        if len(set(directions)) == 1:
            return True, "unanimous"
        
        # Check majority agreement (2/3)
        direction_counts = Counter(directions)
        if max(direction_counts.values()) >= 2:
            return True, "majority"
        
        # Check score convergence
        score_range = max(scores) - min(scores)
        if score_range <= self.convergence_threshold:
            return True, "score_convergence"
        
        return False, "no_convergence"
    
    def _select_discussion_pair(self, agents: Dict[str, ModelAgent]) -> Tuple[str, str]:
        """Select which agent challenges which based on score differences."""
        agent_ids = list(agents.keys())
        
        # Find pair with largest score difference
        max_diff = 0
        challenger_id = agent_ids[0]
        target_id = agent_ids[1]
        
        for i, aid1 in enumerate(agent_ids):
            for aid2 in agent_ids[i+1:]:
                diff = abs(agents[aid1].current_score - agents[aid2].current_score)
                if diff > max_diff:
                    max_diff = diff
                    # The one with more extreme score challenges
                    if abs(agents[aid1].current_score) > abs(agents[aid2].current_score):
                        challenger_id = aid1
                        target_id = aid2
                    else:
                        challenger_id = aid2
                        target_id = aid1
        
        return challenger_id, target_id

    # ==========================================================================
    # SAVING AND LOGGING METHODS
    # ==========================================================================
    
    def _save_individual_results(self, model_name: str, results: List[Dict], articles: List[tuple]):
        """Save individual model results with ground truth."""
        logger.info(f"Saving {model_name} results...")
        
        # Add ground truth to results
        enhanced_results = []
        correct_count = 0
        total_with_truth = 0
        
        for result in results:
            article_idx = result["article_id"]
            if article_idx < len(articles):
                article_data = articles[article_idx][0]
                ground_truth_text, has_ground_truth = get_ground_truth_text(article_data, self.dataset_type)
                
                enhanced_result = result.copy()
                enhanced_result['ground_truth'] = ground_truth_text
                enhanced_result['has_ground_truth'] = has_ground_truth
                
                if has_ground_truth:
                    is_correct = result['direction'].lower() == ground_truth_text
                    enhanced_result['correct'] = is_correct
                    if is_correct:
                        correct_count += 1
                    total_with_truth += 1
                
                enhanced_results.append(enhanced_result)
        
        # Calculate accuracy
        accuracy = (correct_count / total_with_truth * 100) if total_with_truth > 0 else 0
        
        # Save to file
        output_data = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'total_articles': len(results),
            'accuracy': round(accuracy, 2),
            'correct_predictions': correct_count,
            'articles_with_truth': total_with_truth,
            'results': enhanced_results
        }
        
        output_file = self.individual_models_dir / f"{model_name}_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {model_name} results: {output_file}")
        if total_with_truth > 0:
            logger.info(f"{model_name} accuracy: {correct_count}/{total_with_truth} ({accuracy:.1f}%)")
    
    def _save_article_results(self, article_id: int, qwen_res: Dict, gptoss_res: Dict, 
                             mistral_res: Dict, final_result: Dict):
        """Save all results for a single article."""
        article_dir = self.output_dir / f"article_{article_id:04d}"
        article_dir.mkdir(exist_ok=True)
        
        # Save individual responses
        with open(article_dir / "qwen_response.json", 'w', encoding='utf-8') as f:
            json.dump(qwen_res, f, indent=2, ensure_ascii=False)
        
        with open(article_dir / "gptoss_response.json", 'w', encoding='utf-8') as f:
            json.dump(gptoss_res, f, indent=2, ensure_ascii=False)
        
        with open(article_dir / "mistral_response.json", 'w', encoding='utf-8') as f:
            json.dump(mistral_res, f, indent=2, ensure_ascii=False)
        
        # Save final decision
        with open(article_dir / "final_decision.json", 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    def _save_discussion_io(self, discussion_dir: Path, round_num: int, io_type: str,
                           from_agent: str, to_agent: str, prompt: str, response: str):
        """Save raw LLM input/output for discussion."""
        # Save prompt
        prompt_file = discussion_dir / f"round_{round_num}_{io_type}_{from_agent}_to_{to_agent}_prompt.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Round: {round_num}\n")
            f.write(f"Type: {io_type}\n")
            f.write(f"From: {from_agent}\n")
            f.write(f"To: {to_agent}\n")
            f.write("=" * 80 + "\n")
            f.write(prompt)
        
        # Save response
        response_file = discussion_dir / f"round_{round_num}_{io_type}_{from_agent}_to_{to_agent}_response.txt"
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Round: {round_num}\n")
            f.write(f"Type: {io_type}\n")
            f.write(f"From: {from_agent}\n")
            f.write(f"To: {to_agent}\n")
            f.write("=" * 80 + "\n")
            f.write(response)
    
    def _save_discussion_summary(self, discussion_dir: Path, agents: Dict[str, ModelAgent],
                                history: List[Dict], converged: bool, convergence_type: str):
        """Save discussion summary."""
        summary = {
            'initial_states': {
                aid: {
                    'score': agent.initial_analysis['score'],
                    'direction': agent.initial_analysis['direction'],
                    'reason': agent.initial_analysis['reason']
                }
                for aid, agent in agents.items()
            },
            'final_states': {
                aid: {
                    'score': agent.current_score,
                    'direction': agent.current_direction,
                    'reason': agent.current_reason,
                    'changes_made': len(agent.discussion_history)
                }
                for aid, agent in agents.items()
            },
            'discussion_rounds': len(history),
            'convergence_achieved': converged,
            'convergence_type': convergence_type,
            'history': history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(discussion_dir / "discussion_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def _save_session_summary(self, results: List[Dict]):
        """Save overall session summary."""
        summary = {
            'session_info': {
                'timestamp': getattr(self, 'session_timestamp', datetime.now().isoformat()),
                'output_directory': str(self.output_dir),
                'models_used': [
                    self.config['vllm']['regular_ensemble']['qwen3']['model_id'],
                    self.config['vllm']['regular_ensemble']['gptoss']['model_id'],
                    self.config['vllm']['regular_ensemble']['mistral']['model_id'],
                ]
            },
            'statistics': self.stats,
            'consensus_breakdown': {
                'unanimous': self.stats['consensus_unanimous'],
                'majority': self.stats['consensus_majority'],
                'discussion_triggered': self.stats['discussion_triggered'],
                'discussion_converged': self.stats['discussion_converged']
            },
            'results': results
        }
        
        # Save with batch info in filename if processing a batch
        if self.batch_start is not None and self.batch_end is not None:
            filename = f"batch_{self.batch_start}_{self.batch_end}_results.json"
        else:
            filename = "session_summary.json"
        
        with open(self.output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Session summary saved to {self.output_dir}")

    def _display_metrics_comparison(self, pre_metrics: Dict, post_metrics: Dict):
        """Display comparison of metrics before and after discussion."""
        logger.info("=" * 80)
        logger.info("METRICS COMPARISON: Before vs After Discussion")
        logger.info("=" * 80)
        
        if pre_metrics and post_metrics:
            # Calculate improvements
            acc_improvement = post_metrics['accuracy'] - pre_metrics['accuracy']
            f1_improvement = post_metrics['macro_f1'] - pre_metrics['macro_f1']
            
            logger.info(f"                     Before Discussion -> After Discussion")
            logger.info(f"Accuracy:            {pre_metrics['accuracy']:.4f} -> {post_metrics['accuracy']:.4f} "
                       f"({'+' if acc_improvement > 0 else '-' if acc_improvement < 0 else '='} {abs(acc_improvement):.4f})")
            logger.info(f"Macro F1:            {pre_metrics['macro_f1']:.4f} -> {post_metrics['macro_f1']:.4f} "
                       f"({'+' if f1_improvement > 0 else '-' if f1_improvement < 0 else '='} {abs(f1_improvement):.4f})")
            logger.info(f"Articles evaluated:  {pre_metrics['total_samples']}")
            
            # Show percentage improvements
            if pre_metrics['accuracy'] > 0:
                acc_pct_change = (acc_improvement / pre_metrics['accuracy']) * 100
                logger.info(f"\nAccuracy Change: {acc_pct_change:+.2f}%")
            if pre_metrics['macro_f1'] > 0:
                f1_pct_change = (f1_improvement / pre_metrics['macro_f1']) * 100
                logger.info(f"F1 Score Change: {f1_pct_change:+.2f}%")
        else:
            logger.info("Metrics comparison not available (missing ground truth)")
        
        logger.info("=" * 80)
    
    def _display_discussion_impact(self, final_results: List[Dict]):
        """Display summary of discussion impact on results."""
        # Collect discussion articles
        discussion_articles = [r for r in final_results if 'discussion_method' in r]
        
        if not discussion_articles:
            logger.info("No articles required discussion")
            return
        
        logger.info("=" * 80)
        logger.info("DISCUSSION IMPACT SUMMARY")
        logger.info("=" * 80)
        
        # Calculate statistics
        total_discussions = len(discussion_articles)
        converged = sum(1 for r in discussion_articles if r.get('convergence_achieved'))
        direction_changes = sum(1 for r in discussion_articles if r.get('direction_changed'))
        avg_score_change = np.mean([r.get('score_change', 0) for r in discussion_articles])
        errors = sum(1 for r in discussion_articles if 'discussion_error' in r)
        
        logger.info(f"Total articles requiring discussion: {total_discussions}")
        logger.info(f"Discussions that reached convergence: {converged} ({converged/total_discussions*100:.1f}%)")
        logger.info(f"Discussions that changed direction: {direction_changes} ({direction_changes/total_discussions*100:.1f}%)")
        logger.info(f"Average score change: {avg_score_change:.3f}")
        logger.info(f"Discussion errors: {errors}")
        
        # Show before/after comparisons for each discussed article
        logger.info("\nDetailed Before/After Comparison:")
        logger.info("-" * 60)
        for article in discussion_articles:
            article_id = article['article_id']
            pre_score = article.get('pre_discussion_score', 0)
            pre_dir = article.get('pre_discussion_direction', 'Unknown')
            final_score = article.get('final_score', pre_score)
            final_dir = article.get('final_direction', pre_dir)
            converged = article.get('convergence_achieved', False)
            conv_type = article.get('convergence_type', 'none')
            
            logger.info(f"Article {article_id:3d}: {pre_dir:6s} ({pre_score:+.2f}) -> {final_dir:6s} ({final_score:+.2f}) "
                       f"| Converged: {converged} ({conv_type})")
            
            if 'discussion_error' in article:
                logger.info(f"              Error: {article['discussion_error']}")
        
        logger.info("=" * 80)
    
    # ==========================================================================
    # EVALUATION METHODS
    # ==========================================================================
    
    def _calculate_evaluation_metrics(self, final_results: List[Dict], articles: List[tuple], 
                                     display_header: bool = True, model_name: str = None):
        """
        Calculate and display evaluation metrics with per-class metrics.
    
        Args:
            final_results: List of prediction results
            articles: List of (article_data, filename) tuples  
            display_header: Whether to display section headers
            model_name: Name of model/configuration being evaluated
        
        Returns:
            Dictionary containing comprehensive metrics or None if no ground truth
        """
        if display_header:
            logger.info("=" * 80)
            if model_name:
                logger.info(f"EVALUATION METRICS - {model_name}")
            else:
                logger.info("FINAL EVALUATION METRICS")
            logger.info("=" * 80)
    
        # Collect predictions and ground truth
        y_true = []
        y_pred = []
        article_ids = []
    
        for result in final_results:
            article_id = result.get('article_id', result.get('article_idx', -1))
            if article_id < len(articles) and article_id >= 0:
                article_data = articles[article_id][0]
                ground_truth_text, has_ground_truth = get_ground_truth_text(article_data, self.dataset_type)
            
                if has_ground_truth:
                    # Convert to numeric classes
                    true_class = {"left": 0, "center": 1, "right": 2}.get(ground_truth_text, -1)
                
                    # Handle different result formats
                    if 'final_direction' in result:
                        pred_direction = result['final_direction']
                    elif 'direction' in result:
                        pred_direction = result['direction']
                    else:
                        continue
                    
                    pred_class = {"left": 0, "center": 1, "right": 2}.get(pred_direction.lower(), -1)
                
                    if true_class != -1 and pred_class != -1:
                        y_true.append(true_class)
                        y_pred.append(pred_class)
                        article_ids.append(article_id)
    
        if len(y_true) > 0:
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        
            # Calculate per-class metrics
            class_report = classification_report(
                y_true, y_pred,
                labels=[0, 1, 2],
                target_names=['Left', 'Center', 'Right'],
                output_dict=True,
                zero_division=0
            )
        
            # Display metrics
            logger.info(f"Total articles evaluated: {len(y_true)}")
            if model_name:
                logger.info(f"{model_name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            else:
                logger.info(f"Ensemble Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            logger.info(f"Macro F1 Score: {macro_f1:.4f}")
            logger.info(f"Weighted F1 Score: {weighted_f1:.4f}")
        
            # Per-class metrics
            logger.info("\nPer-Class Metrics:")
            logger.info("-" * 60)
            logger.info(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            logger.info("-" * 60)
            for class_name in ['Left', 'Center', 'Right']:
                if class_name in class_report:
                    cr = class_report[class_name]
                    logger.info(
                        f"{class_name:<10} "
                        f"{cr['precision']:<10.4f} "
                        f"{cr['recall']:<10.4f} "
                        f"{cr['f1-score']:<10.4f} "
                        f"{int(cr['support']):<10}"
                    )
        
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            logger.info("\nConfusion Matrix:")
            logger.info("       Pred:")
            logger.info("       L   C   R")
            for i, label in enumerate(["L", "C", "R"]):
                if i < len(cm):
                    logger.info(f"True {label}: {cm[i]}")
        
            # Build metrics dictionary with per-class metrics
            metrics = {
                'total_samples': len(y_true),
                'accuracy': float(accuracy),
                'macro_f1': float(macro_f1),
                'weighted_f1': float(weighted_f1),
                'left_precision': float(class_report.get('Left', {}).get('precision', 0)),
                'left_recall': float(class_report.get('Left', {}).get('recall', 0)),
                'left_f1': float(class_report.get('Left', {}).get('f1-score', 0)),
                'left_support': int(class_report.get('Left', {}).get('support', 0)),
                'center_precision': float(class_report.get('Center', {}).get('precision', 0)),
                'center_recall': float(class_report.get('Center', {}).get('recall', 0)),
                'center_f1': float(class_report.get('Center', {}).get('f1-score', 0)),
                'center_support': int(class_report.get('Center', {}).get('support', 0)),
                'right_precision': float(class_report.get('Right', {}).get('precision', 0)),
                'right_recall': float(class_report.get('Right', {}).get('recall', 0)),
                'right_f1': float(class_report.get('Right', {}).get('f1-score', 0)),
                'right_support': int(class_report.get('Right', {}).get('support', 0)),
                'confusion_matrix': cm.tolist() if isinstance(cm, np.ndarray) else cm,
                'article_ids': article_ids
            }
        
            # Only save to file for final metrics
            if display_header and not model_name:
                with open(self.output_dir / "evaluation_metrics.json", 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"\nMetrics saved to: {self.output_dir / 'evaluation_metrics.json'}")
        
            if display_header:
                logger.info("=" * 80)
        
            return metrics
        else:
            logger.info("No ground truth labels available for evaluation")
            if display_header:
                logger.info("=" * 80)
            return None

    def _comprehensive_evaluation(self, final_results: List[Dict], articles: List[tuple]):
        """
        Perform comprehensive evaluation of all models and ensemble performance.
    
        This method evaluates:
        1. Individual model performance
        2. Consensus-only performance (excluding discussion articles)
        3. Discussion articles analysis (pre vs post discussion)
        4. Overall ensemble performance
        5. Comparative analysis (ensemble vs best individual)
        6. Summary statistics
        """
    
        # Determine model names based on ensemble type
        # NOTE: This is the ONLY part that should differ between files
        if hasattr(self, 'qwen_labeler'):
            model_names = ['qwen', 'gptoss', 'mistral']
            ensemble_type = "regular"
        else:
            model_names = ['llama32', 'qwen3', 'mistral']
            ensemble_type = "small"
    
        # 1. Individual Model Performance
        logger.info("\n" + "="*60)
        logger.info("1. INDIVIDUAL MODEL PERFORMANCE")
        logger.info("="*60)
    
        individual_metrics = {}
    
        for model_name in model_names:
            model_file = self.individual_models_dir / f"{model_name}_results.json"
            if model_file.exists():
                with open(model_file, 'r', encoding='utf-8') as f:
                    model_data = json.load(f)
                    model_results = model_data.get('results', [])
                
                    logger.info(f"\n{model_name.upper()} Model:")
                    metrics = self._calculate_evaluation_metrics(
                        model_results, articles, display_header=False, model_name=model_name
                    )
                    if metrics:
                        individual_metrics[model_name] = metrics
    
        # 2. Consensus-Only Performance (excluding discussion articles)
        logger.info("\n" + "="*60)
        logger.info("2. CONSENSUS-ONLY PERFORMANCE (No Discussion Articles)")
        logger.info("="*60)
    
        consensus_results = []
        discussion_article_ids = set()
    
        for result in final_results:
            if 'discussion_method' in result:
                discussion_article_ids.add(result['article_id'])
            else:
                consensus_results.append(result)
    
        consensus_metrics = None
        if consensus_results:
            logger.info(f"\nEvaluating {len(consensus_results)} consensus articles (unanimous + majority):")
        
            # Count unanimous vs majority
            unanimous_count = sum(1 for r in consensus_results if r.get('consensus_type') == 'unanimous')
            majority_count = sum(1 for r in consensus_results if r.get('consensus_type') == 'majority')
        
            logger.info(f"  - Unanimous agreement: {unanimous_count}")
            logger.info(f"  - Majority agreement: {majority_count}")
        
            consensus_metrics = self._calculate_evaluation_metrics(
                consensus_results, articles, display_header=False, model_name="Consensus-Only"
            )
        
            if consensus_metrics:
                consensus_metrics['consensus_articles'] = len(consensus_results)
                consensus_metrics['unanimous_count'] = unanimous_count
                consensus_metrics['majority_count'] = majority_count
        else:
            logger.info("No consensus-only articles (all required discussion)")
    
        # 3. Discussion Articles Analysis
        logger.info("\n" + "="*60)
        logger.info("3. DISCUSSION ARTICLES ANALYSIS")
        logger.info("="*60)
    
        discussion_results = [r for r in final_results if 'discussion_method' in r]
    
        pre_metrics = None
        post_metrics = None
        discussion_details = []
    
        if discussion_results:
            logger.info(f"\nAnalyzing {len(discussion_results)} articles that triggered discussion:")
        
            # Separate pre and post discussion predictions
            pre_discussion_results = []
            post_discussion_results = []
        
            for result in discussion_results:
                # Pre-discussion (use averaged initial scores)
                pre_result = {
                    'article_id': result['article_id'],
                    'direction': result.get('pre_discussion_direction', 'Center')
                }
                pre_discussion_results.append(pre_result)
            
                # Post-discussion
                post_result = {
                    'article_id': result['article_id'],
                    'direction': result.get('final_direction', 'Center')
                }
                post_discussion_results.append(post_result)
            
                # Track details for each article
                article_id = result['article_id']
                if article_id < len(articles):
                    article_data = articles[article_id][0]
                    ground_truth_text, has_ground_truth = get_ground_truth_text(article_data, self.dataset_type)
                
                    if has_ground_truth:
                        pre_correct = result.get('pre_discussion_direction', '').lower() == ground_truth_text
                        post_correct = result.get('final_direction', '').lower() == ground_truth_text
                    
                        discussion_details.append({
                            'article_id': article_id,
                            'ground_truth': ground_truth_text.capitalize(),
                            'pre_discussion': result.get('pre_discussion_direction', 'Unknown'),
                            'post_discussion': result.get('final_direction', 'Unknown'),
                            'direction_changed': result.get('direction_changed', False),
                            'score_change': result.get('score_change', 0),
                            'pre_correct': pre_correct,
                            'post_correct': post_correct
                        })
        
            logger.info("\nPre-Discussion Performance:")
            pre_metrics = self._calculate_evaluation_metrics(
                pre_discussion_results, articles, display_header=False, model_name="Pre-Discussion"
            )
        
            logger.info("\nPost-Discussion Performance:")
            post_metrics = self._calculate_evaluation_metrics(
                post_discussion_results, articles, display_header=False, model_name="Post-Discussion"
            )
        
            # Calculate improvement
            if pre_metrics and post_metrics:
                acc_change = post_metrics['accuracy'] - pre_metrics['accuracy']
                f1_change = post_metrics['macro_f1'] - pre_metrics['macro_f1']
            
                logger.info("\nDiscussion Impact:")
                logger.info(f"  Accuracy change: {acc_change:+.4f} ({pre_metrics['accuracy']:.4f} -> {post_metrics['accuracy']:.4f})")
                logger.info(f"  Macro F1 change: {f1_change:+.4f} ({pre_metrics['macro_f1']:.4f} -> {post_metrics['macro_f1']:.4f})")
            
                # Analyze individual changes
                improvement_count = sum(1 for d in discussion_details if not d['pre_correct'] and d['post_correct'])
                degradation_count = sum(1 for d in discussion_details if d['pre_correct'] and not d['post_correct'])
                no_change_count = sum(1 for d in discussion_details if d['pre_correct'] == d['post_correct'])
            
                logger.info(f"\n  Article-level changes:")
                logger.info(f"    - Improved: {improvement_count}")
                logger.info(f"    - Degraded: {degradation_count}")
                logger.info(f"    - No change: {no_change_count}")
        else:
            logger.info("No articles triggered discussion")
    
        # 4. Overall Ensemble Performance
        logger.info("\n" + "="*60)
        logger.info("4. OVERALL ENSEMBLE PERFORMANCE (ALL ARTICLES)")
        logger.info("="*60)
    
        overall_metrics = self._calculate_evaluation_metrics(
            final_results, articles, display_header=False, model_name="Overall Ensemble"
        )
    
        # 5. Comparative Analysis
        logger.info("\n" + "="*60)
        logger.info("5. COMPARATIVE ANALYSIS")
        logger.info("="*60)
    
        if individual_metrics and overall_metrics:
            # Find best individual model
            best_individual = max(individual_metrics.items(), 
                                 key=lambda x: x[1].get('accuracy', 0))
            best_model_name, best_model_metrics = best_individual
        
            logger.info(f"\nBest Individual Model: {best_model_name.upper()}")
            logger.info(f"  Accuracy: {best_model_metrics['accuracy']:.4f}")
            logger.info(f"  Macro F1: {best_model_metrics['macro_f1']:.4f}")
        
            # Compare ensemble to best individual
            acc_improvement = overall_metrics['accuracy'] - best_model_metrics['accuracy']
            f1_improvement = overall_metrics['macro_f1'] - best_model_metrics['macro_f1']
        
            logger.info(f"\nEnsemble vs Best Individual:")
            logger.info(f"  Accuracy: {overall_metrics['accuracy']:.4f} vs {best_model_metrics['accuracy']:.4f} "
                       f"({'+' if acc_improvement > 0 else '-' if acc_improvement < 0 else '='} {abs(acc_improvement):.4f})")
            logger.info(f"  Macro F1: {overall_metrics['macro_f1']:.4f} vs {best_model_metrics['macro_f1']:.4f} "
                       f"({'+' if f1_improvement > 0 else '-' if f1_improvement < 0 else '='} {abs(f1_improvement):.4f})")
        
            if acc_improvement > 0:
                logger.info(f"\n[OK] Ensemble OUTPERFORMS best individual model by {acc_improvement:.4f} accuracy")
            elif acc_improvement < 0:
                logger.info(f"\nWARNING: Ensemble UNDERPERFORMS best individual model by {abs(acc_improvement):.4f} accuracy")
            else:
                logger.info(f"\n= Ensemble matches best individual model performance")
    
        # 6. Summary Statistics
        logger.info("\n" + "="*60)
        logger.info("6. SUMMARY STATISTICS")
        logger.info("="*60)
    
        logger.info(f"\nTotal articles processed: {self.stats['total_articles']}")
        logger.info(f"Consensus reached: {self.stats['consensus_unanimous'] + self.stats['consensus_majority']}")
        logger.info(f"  - Unanimous: {self.stats['consensus_unanimous']}")
        logger.info(f"  - Majority: {self.stats['consensus_majority']}")
        logger.info(f"Discussion triggered: {self.stats['discussion_triggered']}")
        logger.info(f"Discussion converged: {self.stats['discussion_converged']}")
        logger.info(f"Articles skipped: {self.stats['articles_skipped']}")
        logger.info(f"\nModel errors:")
        for model, errors in self.stats['model_errors'].items():
            if errors > 0:
                logger.info(f"  - {model}: {errors}")
    
        # Save comprehensive evaluation results
        comprehensive_results = {
            'session_info': {
                'session_dir': str(self.output_dir),
                'evaluation_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'statistics': self.stats
            },
            'individual_models': individual_metrics,
            'consensus_only': consensus_metrics,
            'discussion_analysis': {
                'pre_discussion': pre_metrics,
                'post_discussion': post_metrics,
                'improvement_count': improvement_count if 'improvement_count' in locals() else 0,
                'degradation_count': degradation_count if 'degradation_count' in locals() else 0,
                'no_change_count': no_change_count if 'no_change_count' in locals() else 0,
                'accuracy_change': (post_metrics['accuracy'] - pre_metrics['accuracy']) if pre_metrics and post_metrics else 0,
                'details': discussion_details
            },
            'overall_final': overall_metrics
        }
    
        # Use different filename based on ensemble type
        if ensemble_type == "small":
            output_file = self.output_dir.parent / f"small_ensemble_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            output_file = self.output_dir.parent / f"ensemble_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
    
        logger.info(f"\nComprehensive evaluation saved to: {output_file}")
        logger.info("=" * 80)
    
    def _validate_results(self, qwen_results: List, gptoss_results: List, mistral_results: List) -> bool:
        """Validate that all models processed the same articles."""
        if len(qwen_results) != len(gptoss_results) or len(gptoss_results) != len(mistral_results):
            logger.error(f"Result count mismatch: Qwen={len(qwen_results)}, "
                        f"GPT-OSS={len(gptoss_results)}, Mistral={len(mistral_results)}")
            return False
        
        if len(qwen_results) == 0:
            logger.error("No articles were successfully processed")
            return False
        
        return True


# ==============================================================================
# ARTICLE LOADING FUNCTIONS
# ==============================================================================

def load_balanced_dataset_articles(data_dir: Path) -> Tuple[List[tuple], str]:
    """
    Load articles from a balanced dataset directory.
    
    Args:
        data_dir: Path to balanced dataset directory
        
    Returns:
        Tuple of (articles list, dataset_type)
    """
    manifest_path = data_dir / 'dataset_manifest.json'
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found at {manifest_path}")
    
    # Load manifest
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    dataset_type = manifest['dataset_info']['dataset_type']
    total_articles = manifest['dataset_info']['total_articles']
    
    logger.info(f"Loading balanced dataset:")
    logger.info(f"  Type: {dataset_type}")
    logger.info(f"  Total articles: {total_articles}")
    # Custom datasets don't have articles_per_class field
    if 'articles_per_class' in manifest['dataset_info']:
        logger.info(f"  Articles per class: {manifest['dataset_info']['articles_per_class']}")
    logger.info(f"  Created: {manifest['dataset_info']['creation_time']}")
    
    # Load articles
    articles = []
    for article_info in manifest['articles']:
        article_path = data_dir / article_info['filename']
        with open(article_path, 'r', encoding='utf-8') as f:
            article_data = json.load(f)
        articles.append((article_data, article_info['filename']))
    
    logger.info(f"Successfully loaded {len(articles)} articles from balanced dataset")
    return articles, dataset_type


def find_balanced_dataset(config: Dict, dataset_type: str) -> Optional[Path]:
    """
    Find the balanced dataset for a specific dataset type.
    
    Args:
        config: Configuration dictionary
        dataset_type: Dataset type ('baly', 'budak', or 'ad_fontes')
    
    Returns:
        Path to the balanced dataset or None if not found
    """
    if not dataset_type:
        raise ValueError("Dataset type must be specified")
    
    balanced_base = Path(config['dirs']['balanced_datasets'])
    if not balanced_base.exists():
        return None
    
    # Fixed path for each dataset type
    if dataset_type == 'custom':
        balanced_dir = balanced_base / "custom_100_per_outlet"
    else:
        balanced_dir = balanced_base / f"balanced_{dataset_type}"
    
    # Check if balanced dataset exists and has valid manifest
    if balanced_dir.exists() and (balanced_dir / 'dataset_manifest.json').exists():
        logger.info(f"Found balanced dataset for {dataset_type}: {balanced_dir}")
        return balanced_dir
    
    return None


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

async def main(data_dir: str = None, dataset: str = None, n_samples: int = 1000, use_original: bool = False,
               batch_start: int = 0, batch_end: int = None, output_dir: str = None, skip_evaluation: bool = False):
    """
    Main execution function.
    
    Args:
        data_dir: Path to specific balanced dataset directory (overrides auto-detection)
        dataset: Dataset to use when loading original data ('baly', 'budak', or 'ad_fontes')
        n_samples: Number of articles to process (only for original data)
        use_original: Force use of original unbalanced data
    """
    config = get_config()
    
    # Determine data source
    if data_dir:
        # User specified a specific directory
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Specified data directory not found: {data_path}")
        
        # Check if it's a balanced dataset
        if (data_path / 'dataset_manifest.json').exists():
            logger.info(f"Using specified balanced dataset: {data_path}")
            articles, dataset_type = load_balanced_dataset_articles(data_path)
        else:
            raise ValueError(f"Directory {data_path} is not a valid balanced dataset (missing manifest)")
    
    elif use_original:
        # Original (unbalanced) datasets are not shipped with this repo.
        raise NotImplementedError(
            "--use-original is not supported in this repository. The original "
            "unbalanced Baly/Budak/Ad Fontes corpora belong to the legacy "
            "MediaBiasDetection project. Create a balanced dataset with "
            "tools/create_balanced_dataset.py and run without --use-original."
        )
    
    else:
        # Default: Must specify dataset type and load its balanced version
        if not dataset:
            logger.error("\nERROR: No dataset specified!")
            logger.error("\nYou must specify which dataset to use with --dataset")
            logger.error("\nExamples:")
            logger.error("  python ensemble_multi_model.py --dataset baly")
            logger.error("  python ensemble_multi_model.py --dataset ad_fontes")
            logger.error("  python ensemble_multi_model.py --dataset budak")
            raise ValueError("Dataset type must be specified with --dataset parameter")
        
        dataset_type = dataset.lower()
        
        # Look for the balanced dataset for this type
        balanced_dataset_path = find_balanced_dataset(config, dataset_type)
        
        if balanced_dataset_path:
            logger.info(f"Using balanced dataset for {dataset_type}: {balanced_dataset_path}")
            articles, _ = load_balanced_dataset_articles(balanced_dataset_path)
        else:
            # No balanced dataset exists - show error and exit
            expected_path = Path(config['dirs']['balanced_datasets']) / f'balanced_{dataset_type}'
            logger.error(f"\nERROR: No balanced dataset found for '{dataset_type}'")
            logger.error(f"Expected location: {expected_path}")
            logger.error("\nTo create the balanced dataset, run:")
            logger.error(f"  python create_balanced_dataset.py --dataset {dataset_type} --n_samples 1000")
            logger.error("\nOr to use original unbalanced data, run:")
            logger.error(f"  python ensemble_multi_model.py --use-original --dataset {dataset_type} --n_samples {n_samples}")
            raise FileNotFoundError(f"Balanced dataset for '{dataset_type}' not found. Create it first or use --use-original flag.")
    
    # Apply batch slicing if specified
    if batch_end is not None:
        original_count = len(articles)
        articles = articles[batch_start:batch_end]
        logger.info(f"Processing batch: articles {batch_start}-{batch_end} (out of {original_count} total)")
    elif batch_start > 0:
        articles = articles[batch_start:]
        logger.info(f"Processing articles from index {batch_start} onwards")
    
    # Initialize ensemble
    ensemble = EnsembleMultiModelDetector(
        config, 
        batch_size=6, 
        dataset_type=dataset_type,
        output_dir=output_dir,
        batch_start=batch_start,
        batch_end=batch_end
    )
    
    # Process articles - skip evaluation if requested
    if skip_evaluation:
        # Temporarily disable comprehensive evaluation
        original_method = ensemble._comprehensive_evaluation
        ensemble._comprehensive_evaluation = lambda *args, **kwargs: None
    
    results = await ensemble.process_articles(articles)
    
    # Save article list with the results
    article_list_path = ensemble.output_dir / 'article_list.json'
    article_list = {
        'dataset_type': dataset_type,
        'total_articles': len(articles),
        'articles': [
            {
                'index': i,
                'filename': articles[i][1] if i < len(articles) else f'article_{i}'
            }
            for i in range(len(results))
        ]
    }
    with open(article_list_path, 'w', encoding='utf-8') as f:
        json.dump(article_list, f, indent=2)
    logger.info(f"Article list saved to: {article_list_path}")
    
    print(f"\nDataset type: {dataset_type.upper()}")
    print(f"Processed {len(results)} articles")
    print(f"Output directory: {ensemble.output_dir}")
    print(f"Statistics: {ensemble.stats}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Multi-model ensemble bias detection with balanced dataset support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use balanced dataset for baly
  python ensemble_multi_model.py --dataset baly
  
  # Use balanced dataset for ad_fontes
  python ensemble_multi_model.py --dataset ad_fontes
  
  # Use specific balanced dataset directory
  python ensemble_multi_model.py --data-dir data/balanced_datasets/balanced_baly
  
  # Use original unbalanced data
  python ensemble_multi_model.py --use-original --dataset baly --n_samples 500
        """
    )
    
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Path to specific balanced dataset directory')
    parser.add_argument('--use-original', action='store_true',
                       help='Use original unbalanced data instead of balanced dataset')
    parser.add_argument('--dataset', type=str, default=None,
                       choices=['baly', 'budak', 'ad_fontes', 'custom'],
                       help='Dataset type to use (required unless --data-dir is specified)')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of articles (only for original data)')
    parser.add_argument('--batch-start', type=int, default=0,
                       help='Starting index for batch processing')
    parser.add_argument('--batch-end', type=int, default=None,
                       help='Ending index for batch processing (None = process all)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (shared across batches)')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip comprehensive evaluation (for batch processing)')
    
    args = parser.parse_args()
    
    asyncio.run(main(
        data_dir=args.data_dir,
        dataset=args.dataset,
        n_samples=args.n_samples,
        use_original=args.use_original,
        batch_start=args.batch_start,
        batch_end=args.batch_end,
        output_dir=args.output_dir,
        skip_evaluation=args.skip_evaluation
    ))