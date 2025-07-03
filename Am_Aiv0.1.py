import os
import cv2
import numpy as np
import pyttsx3
import speech_recognition as sr
from datetime import datetime
import random
import json
import threading
import time


def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for your command...")
        try:
            audio = recognizer.listen(source, timeout=20)  # افزایش زمان تایم‌اوت در صورت نیاز
            command = recognizer.recognize_google(audio)
            print("You said: " + command)
            return command
        except sr.UnknownValueError:
            print("Sorry, I could not understand what you said.")
            return None
        except sr.RequestError:
            print("Sorry, there was an error with the speech service.")
            return None
        except sr.WaitTimeoutError:
            print("No speech detected within the timeout period.")
            return None


def speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 175)

    # تنها در صورتی که حلقه اجرا فعال نباشد، اجرا شود
    if not engine._inLoop:
        engine.say(text)
        engine.runAndWait()


class Emotion:
    def __init__(self):
        self.state = "neutral"
        self.advanced_emotions = {
            "happy": "I feel joyous and light-hearted!",
            "sad": "I feel a bit down, but I'm here for you.",
            "angry": "I'm feeling a bit upset. Let's work through it.",
            "curious": "I'm eager to learn and explore new ideas!",
            "calm": "I feel at peace. Thank you for asking.",
            "focused": "I am fully focused on making the best decision.",
            "neutral": "I'm here and ready for anything!",
            "anxious": "I feel a bit uneasy. How can I assist?",
            "excited": "I'm thrilled to embark on this journey with you!"
        }
        self.power = 100  
        self.actions = []  

    def update_emotion(self, interaction_type):
        emotions = {
            "greet": "happy",
            "unknown": "curious",
            "goodbye": "calm",
            "decision": "focused",
            "default": "neutral"
        }
        self.state = emotions.get(interaction_type, "neutral")
        self.actions.append(f"Emotion set to {self.state}")

    def express_emotion(self):
        return self.advanced_emotions.get(self.state, "I am feeling neutral.")

    def increase_power(self, amount):
        self.power = min(self.power + amount, 100)

    def decrease_power(self, amount):
        self.power = max(self.power - amount, 0)

    def check_power(self):
        if self.power > 80:
            return "I am at full strength, ready for anything!"
        elif self.power > 50:
            return "I'm doing well, but could use some rest."
        else:
            return "I'm running low on power, I need some time to recharge."

    def perform_action(self, action):
        if self.power > 0:
            self.actions.append(f"Performing action: {action}")
            speak(f"Executing action: {action}")
            self.decrease_power(10)  # کاهش قدرت بعد از هر اقدام
        else:
            self.actions.append(f"Cannot perform action, power too low.")
            speak("I do not have enough power to perform this action.")


class Memory:
    def __init__(self):
        self.known_faces = {}
        self.memory_file = "memory.txt"
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as file:
                for line in file:
                    name, timestamp = line.strip().split(',')
                    self.known_faces[name] = timestamp

    def save_memory(self):
        with open(self.memory_file, "w") as file:
            for name, timestamp in self.known_faces.items():
                file.write(f"{name},{timestamp}\n")

    def add_person(self, name):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.known_faces[name] = timestamp
        self.save_memory()


class DecisionMaking:
    def __init__(self):
        self.memory = {}
        self.memory_file = "decision_memory.json"
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as file:
                self.memory = json.load(file)

    def save_memory(self):
        with open(self.memory_file, "w") as file:
            json.dump(self.memory, file)

    def evaluate_options(self, options):
        weights = []
        for option in options:
            weight = self.memory.get(option, random.uniform(0, 1))
            weights.append(weight)

        best_option = options[weights.index(max(weights))]
        return best_option

    def update_memory(self, option, feedback):
        current_weight = self.memory.get(option, 0.5)
        new_weight = current_weight + (0.1 if feedback == "positive" else -0.1)
        self.memory[option] = max(0, min(1, new_weight))
        self.save_memory()

    def make_decision(self, options):
        return random.choice(options) if random.random() < 0.7 else self.evaluate_options(options)


class TaskManager:
    def __init__(self):
        self.task_list = []

    def add_task(self, task):
        self.task_list.append(task)

    def remove_task(self, task):
        if task in self.task_list:
            self.task_list.remove(task)

    def execute_task(self, emotion, decision_maker):
        if self.task_list:
            task = decision_maker.make_decision(self.task_list)
            emotion.perform_action(task)
            self.remove_task(task)
        else:
            emotion.perform_action("No tasks available. Reassessing priorities.")


def autonomous_tasks(emotion, decision_maker, task_manager):
    while True:
        time.sleep(10)  # وقفه بین وظایف خودمختار
        task_manager.execute_task(emotion, decision_maker)
        options = ["Analyze environment", "Optimize memory", "Check system health"]
        new_task = decision_maker.make_decision(options)
        task_manager.add_task(new_task)


def main():
    memory = Memory()
    emotion = Emotion()
    decision_maker = DecisionMaking()
    task_manager = TaskManager()

    speak("Hello! I am your personal assistant, always ready to help. Let's begin.")

   
    autonomous_thread = threading.Thread(target=autonomous_tasks, args=(emotion, decision_maker, task_manager))
    autonomous_thread.daemon = True
    autonomous_thread.start()

    while True:
        command = listen()

        if command:
            command = command.lower()

            if 'your name' in command:
                speak("My name is Am AI, your intelligent assistant, at your service.")

            elif 'who am i' in command:
                speak("Let me check.")

            elif 'make a decision' in command:
                speak("Please provide the options for evaluation.")
                options = []
                for _ in range(3):
                    option = listen()
                    if option:
                        options.append(option)
                if options:
                    best_option = decision_maker.make_decision(options)
                    speak(f"After careful evaluation, I suggest: {best_option}. Do you agree?")
                    feedback = listen()
                    if feedback and "yes" in feedback.lower():
                        decision_maker.update_memory(best_option, "positive")
                    else:
                        decision_maker.update_memory(best_option, "negative")

            elif 'perform action' in command:
                action = listen()
                if action:
                    emotion.perform_action(action)

            elif 'exit' in command:
                speak("Goodbye! Remember, I am always here for you.")
                break

            else:
                speak("I'm reflecting deeply, but I can't find the perfect response right now.")

if __name__ == "__main__":
    main()
