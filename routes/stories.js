// routes/stories.js
const express = require('express');
const mongoose = require('mongoose');
const Story = require('../models/Story'); // Adjust the path as necessary

const router = express.Router();

// GET all stories
router.get('/', async (req, res) => {
    console.log('[Stories Route] GET request received for all stories');
    try {
        const stories = await Story.find();
        console.log(`[Stories Route] Fetched ${stories.length} stories`);
        res.json(stories);
    } catch (error) {
        console.error('[Stories Route] Error fetching stories:', error);
        res.status(500).json({ message: 'Error fetching stories' });
    }
});

// GET individual story by ID
router.get('/:id', async (req, res) => {
    const { id } = req.params;
    console.log(`[Stories Route] GET request received for story with ID: ${id}`);

    try {
        console.log(`[Stories Route] Checking if ID is valid ObjectId: ${id}`);
        if (!mongoose.Types.ObjectId.isValid(id)) {
            console.log(`[Stories Route] Invalid ObjectId: ${id}`);
            return res.status(400).json({ message: 'Invalid story ID format' });
        }

        console.log(`[Stories Route] Searching for story with ID: ${id}`);
        const story = await Story.findById(id);
        console.log(`[Stories Route] Story search result:`, story);

        if (!story) {
            console.log(`[Stories Route] No story found with ID: ${id}`);
            return res.status(404).json({ message: 'Story not found' });
        }

        console.log(`[Stories Route] Successfully fetched story with ID: ${id}`);
        res.json(story);
    } catch (error) {
        console.error(`[Stories Route] Error fetching story with ID ${id}:`, error);
        res.status(500).json({ message: 'Error fetching story' });
    }
});

// Optional: POST route to store results (answers) from users (if you need it later)
router.post('/result', async (req, res) => {
    const { storyId, answers } = req.body;

    try {
        // Assuming 'answers' is an array of user responses for the questions in a story
        console.log(`[Stories Route] Storing answers for story with ID: ${storyId}`, answers);
        
        // Logic to store answers (or process them)
        // For example, you could save this to a database or send it for further processing
        res.status(200).json({ message: 'Answers received successfully', data: answers });
    } catch (error) {
        console.error('[Stories Route] Error storing answers:', error);
        res.status(500).json({ message: 'Error storing answers' });
    }
});

module.exports = router;
