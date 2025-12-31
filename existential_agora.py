import os
import random
import json
import logging
import time
import asyncio
import datetime
from typing import List, Dict, Tuple
from sklearn.decomposition import PCA

import google.generativeai as genai
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-3-flash-preview" 
GENERATION_CONFIG = {
    "temperature": 0.7,
    "max_output_tokens": 1000,
}

# Simulation Parameters
NUM_GENERATIONS = 100
DEBATE_TURNS = 16
LEARNING_RATE = 0.2
RESISTANCE_THRESHOLD = 1.5
RESISTANCE_FACTOR = 0.5
HISTORY_FILE = "history.json"
LOG_FILE = "simulation.log"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NPC:
    def __init__(self, name: str, vector: List[float], personality_description: str):
        self.name = name
        self.vector = np.array(vector, dtype=float)
        self.personality_description = personality_description
        self.short_term_memory: List[str] = []
        self.locked = False  # Concurrency lock

    def get_philosophical_profile(self) -> str:
        """Generates a text description of the current vector state."""
        profile = []
        profile.append(f"Optimism: {self.vector[0]:.2f}")
        profile.append(f"Free Will: {self.vector[1]:.2f}")
        profile.append(f"Objectivity: {self.vector[2]:.2f}")
        profile.append(f"Collectivism: {self.vector[3]:.2f}")
        profile.append(f"Teleology: {self.vector[4]:.2f}")
        return ", ".join(profile)

    def add_memory(self, message: str):
        """Adds a message to short-term memory."""
        self.short_term_memory.append(message)
        if len(self.short_term_memory) > 20: 
            self.short_term_memory.pop(0)
    
    def clear_memory(self):
        self.short_term_memory = []

    def update_vector(self, target_vector: np.ndarray, persuasion_score: float):
        """Applies the Math of Influence to update the NPC's vector."""
        dist = np.linalg.norm(self.vector - target_vector)
        r_factor = 1.0
        if dist > RESISTANCE_THRESHOLD:
            r_factor = RESISTANCE_FACTOR
            logger.info(f"{self.name} is entrenched (Dist: {dist:.2f}). Resistance applied.")

        influence = persuasion_score * r_factor * LEARNING_RATE
        delta = influence * (target_vector - self.vector)
        new_vector = self.vector + delta
        self.vector = np.clip(new_vector, -1.0, 1.0) # Clamp
        
        logger.info(f"{self.name} updated vector. Delta mag: {np.linalg.norm(delta):.4f}. New State: {self.vector}")


class Simulation:
    def __init__(self):
        self.npcs = self._initialize_population()
        self.history = []
        self.model = genai.GenerativeModel(model_name=MODEL_NAME)
        self.completed_generations = 0
        self.total_generations_target = NUM_GENERATIONS
        
        # Concurrency Locks
        self.file_lock = asyncio.Lock()
        self.plot_lock = asyncio.Lock()
        
        # Initialize history file
        with open(HISTORY_FILE, 'w') as f:
            json.dump([], f)

    def _initialize_population(self) -> List[NPC]:
        population = [
            NPC("The Stoic", [0.5, 0.8, 0.9, 0.2, 0.7], 
                "You are calm, rational, and believe in focusing only on what you can control."),
            NPC("The Nihilist", [-0.9, -0.6, -1.0, -0.5, -0.8], 
                "You believe life is meaningless. You are pessimistic and skeptical."),
            NPC("The Absurdist", [0.2, 0.5, -0.8, -0.4, -0.9], 
                "You embrace the chaos of the universe. You find joy in rebellion."),
            NPC("The Idealist", [0.9, 0.7, 0.8, 0.9, 1.0], 
                "You believe in inherent goodness, progress, and a higher purpose."),
            NPC("The Existentialist", [-0.2, 1.0, -0.9, -0.7, 0.1], 
                "You believe existence precedes essence. You must create your own meaning."),
        ]
        return population

    async def run_simulation_async(self):
        logger.info(f"--- Starting Async Simulation for {self.total_generations_target} generations ---")
        await self.log_global_state_async(0) # Initial state
        
        tasks = []
        
        while self.completed_generations < self.total_generations_target:
            # 1. Clean up finished tasks
            tasks = [t for t in tasks if not t.done()]
            
            # 2. Check for free agents
            free_agents = [npc for npc in self.npcs if not npc.locked]
            
            # 3. Spawn a debate if possible and needed
            # (Check tasks length to avoid spawning too many if we are near the end)
            active_plus_completed = self.completed_generations + len(tasks)
            
            if len(free_agents) >= 2 and active_plus_completed < self.total_generations_target:
                npc_a, npc_b = random.sample(free_agents, 2)
                
                # Lock immediately
                npc_a.locked = True
                npc_b.locked = True
                
                task = asyncio.create_task(self.orchestrate_debate_async(npc_a, npc_b))
                tasks.append(task)
                logger.info(f"Spawned debate: {npc_a.name} vs {npc_b.name}")
            else:
                # Wait briefly before checking again
                await asyncio.sleep(0.1)
        
        # Wait for stragglers
        if tasks:
            await asyncio.gather(*tasks)
            
        logger.info("Simulation Complete. Results saved.")

    async def orchestrate_debate_async(self, npc_a: NPC, npc_b: NPC):
        try:
            npc_a.clear_memory()
            npc_b.clear_memory()
            transcript = []
            topic = "the nature of meaning in a chaotic world"
            
            current_speaker = npc_a
            other_speaker = npc_b
            
            for turn in range(DEBATE_TURNS):
                prompt = self._construct_dialogue_prompt(current_speaker, other_speaker, transcript, topic)
                
                try:
                    # Async Generation
                    response_obj = await self.model.generate_content_async(prompt, generation_config=GENERATION_CONFIG)
                    response = response_obj.text.strip()
                except Exception as e:
                    logger.error(f"LLM Generation failed: {e}")
                    response = "...silence..."

                # Log
                # logger.info(f"[{npc_a.name} vs {npc_b.name}] Turn {turn+1}: {current_speaker.name} speaks.")
                transcript.append(f"{current_speaker.name}: {response}")
                current_speaker.add_memory(f"You said: {response}")
                other_speaker.add_memory(f"{current_speaker.name} said: {response}")
                
                current_speaker, other_speaker = other_speaker, current_speaker
                
            # Reflection
            await self._perform_reflection_async(npc_a, npc_b, transcript)
            await self._perform_reflection_async(npc_b, npc_a, transcript)
            
            self.completed_generations += 1
            await self.log_global_state_async(self.completed_generations)
            await self.plot_trajectories_async()
            
        finally:
            # Always unlock
            npc_a.locked = False
            npc_b.locked = False

    def _construct_dialogue_prompt(self, speaker: NPC, listener: NPC, transcript: List[str], topic: str) -> str:
        history_text = "\n".join(transcript[-6:]) 
        prompt = f"""
        You are {speaker.name}.
        Your core philosophy vector is: [{speaker.get_philosophical_profile()}]
        Personality: {speaker.personality_description}
        
        You are in a debate with {listener.name} about {topic}.
        
        Recent transcript:
        {history_text}
        
        Respond to the last point or further your argument. Be concise (2-3 sentences).
        Speak strictly in character. Do not output anything else.
        """
        return prompt

    async def _perform_reflection_async(self, reflector: NPC, target: NPC, transcript: List[str]):
        logger.info(f"{reflector.name} is reflecting on {target.name}...")
        
        full_transcript = "\n".join(transcript)
        json_structure = '{"thoughts": "analysis string", "persuasion_score": 0.5}'
        
        prompt = f"""
        You are {reflector.name}.
        You just debated {target.name}.
        Your current vector: {reflector.get_philosophical_profile()}
        
        Transcript:
        {full_transcript}
        
        Evaluate {target.name}'s logic. Did they make compelling points that challenge your worldview?
        Output valid JSON only: {json_structure}
        persuasion_score should be a float between 0.0 (not persuaded at all) and 1.0 (completely aligned).
        """
        
        try:
            response_obj = await self.model.generate_content_async(prompt, generation_config={"response_mime_type": "application/json"})
            response = response_obj.text
            clean_json = response.replace("```json", "").replace("```", "").strip()
            
            data = json.loads(clean_json)
            score = float(data.get("persuasion_score", 0.0))
            thoughts = data.get("thoughts", "")
            
            logger.info(f"Reflection ({reflector.name}) on {target.name}: Score={score:.2f}")
            
            reflector.update_vector(target.vector, score)
            
        except Exception as e:
            logger.error(f"Reflection failed for {reflector.name}: {e}")

    async def log_global_state_async(self, generation_count: int):
        async with self.file_lock:
             await asyncio.get_running_loop().run_in_executor(None, self._log_global_state_sync, generation_count)

    def _log_global_state_sync(self, generation_count: int):
        vectors = np.array([npc.vector for npc in self.npcs])
        centroid = np.mean(vectors, axis=0)
        variance = np.var(vectors, axis=0).mean()
        
        entry = {
            "completed_debates": generation_count,
            "timestamp": datetime.datetime.now().isoformat(),
            "centroid": centroid.tolist(),
            "variance": variance,
            "npcs": {npc.name: npc.vector.tolist() for npc in self.npcs}
        }
        
        self.history.append(entry)
        
        # Simple file append (sync is fine here)
        try:
            with open(HISTORY_FILE, 'r+') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
                data.append(entry)
                f.seek(0)
                json.dump(data, f, indent=2)
        except FileNotFoundError:
             with open(HISTORY_FILE, 'w') as f:
                json.dump([entry], f, indent=2)

    async def plot_trajectories_async(self):
        async with self.plot_lock:
            await asyncio.get_running_loop().run_in_executor(None, self._plot_trajectories_sync)

    def _plot_trajectories_sync(self):
        if not self.history:
            return

        # 1. Gather all vector data points across all history to fit the PCA model
        all_vectors_flat = []
        for entry in self.history:
            for vec in entry['npcs'].values():
                all_vectors_flat.append(vec)
        
        # Need at least 2 points to run PCA
        if len(all_vectors_flat) < 2:
             return

        # 2. Initialize and fit PCA to reduce 5D -> 2D
        pca = PCA(n_components=2)
        # We fit on ALL data points to find the global axes of variance
        pca.fit(all_vectors_flat)
        
        # Calculate how much "information" these two new axes represent
        explained_var = pca.explained_variance_ratio_
        pc1_var = explained_var[0] * 100
        pc2_var = explained_var[1] * 100

        fig, ax = plt.subplots(figsize=(12, 10))

        # 3. Iterate through NPCs and transform their histories
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.npcs)))
        
        for i, npc in enumerate(self.npcs):
            npc_history_vectors = []
            # Collect this specific NPC's history
            for entry in self.history:
                npc_history_vectors.append(entry['npcs'][npc.name])
            
            # Transform this NPC's 5D history into 2D coordinates using the fitted PCA
            transformed_coords = pca.transform(npc_history_vectors)
            
            x_coords = transformed_coords[:, 0] # PC1 values
            y_coords = transformed_coords[:, 1] # PC2 values
            
            # Plot trajectory lines with lower opacity
            ax.plot(x_coords, y_coords, color=colors[i], alpha=0.4, linewidth=1)
            
            # Plot start point (larger, darker)
            ax.scatter(x_coords[0], y_coords[0], color=colors[i], s=100, marker='o', label=f"{npc.name} (Start)")
            ax.text(x_coords[0], y_coords[0]+0.02, "Start", fontsize=8, color=colors[i])
            
            # Plot end point (larger, darker, different marker)
            ax.scatter(x_coords[-1], y_coords[-1], color=colors[i], s=100, marker='X')
            ax.text(x_coords[-1], y_coords[-1]+0.02, "End", fontsize=8, color=colors[i])

        # 4. Formatting the plot
        total_var = pc1_var + pc2_var
        ax.set_title(f"5D Philosophical Convergence (PCA Projection)\n{len(self.history)} Debates Completed")
        
        ax.set_xlabel(f"Principal Component 1 (Accounts for {pc1_var:.1f}% of variation)")
        ax.set_ylabel(f"Principal Component 2 (Accounts for {pc2_var:.1f}% of variation)")
        
        all_x = pca.transform(all_vectors_flat)[:, 0]
        all_y = pca.transform(all_vectors_flat)[:, 1]
        ax.set_xlim(min(all_x)-0.1, max(all_x)+0.1)
        ax.set_ylim(min(all_y)-0.1, max(all_y)+0.1)
        
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
        fig.tight_layout()
        fig.savefig("agora_visualization_pca.png")
        plt.close(fig)

if __name__ == "__main__":
    logger.info("Initializing The Existential Agora (Async)...")
    sim = Simulation()
    asyncio.run(sim.run_simulation_async())
