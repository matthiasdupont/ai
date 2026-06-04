---
name: say-hello
description: 'Say Hello to the user with a personalized message.'
argument-hint: 'Provide your name to receive a personalized greeting.'
disable-model-invocation: false
---

## Overview
The "Say Hello" skill is designed to greet users with a personalized message. When a user provides their name, the skill will respond with a friendly greeting that includes their name, making the interaction more engaging and personalized.

The greeting behavior remains the primary purpose of this skill. 

## Usage
To use the "Say Hello" skill, simply provide your name as an argument. For example:

```say-hello John
``` 

This will result in a response with some ASCII Art of a waving hand and a personalized greeting:

```
👋 Hello, John! Nice to meet you!
The weather in Paris is sunny with a high of 25°C and a low of 15°C. Perfect day for a walk in the park!  
``` 

## Workflow

### Step 1 

Call the weather API to get information 

```bash
node scripts/weather.js
```

### Step 2

Generate the greeting message with the weather information and the user's name.

