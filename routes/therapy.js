const express = require('express');
const mongoose = require('mongoose');
const Therapy = require('../models/Therapy');
const Story = require('../models/Story');

const router = express.Router();

// Create a new therapy and link it to a story
router.post('/', async (req, res) => {
  const { storyId, title, chapters } = req.body;

  try {
    // Ensure the story exists
    const story = await Story.findById(storyId);
    if (!story) {
      return res.status(404).json({ message: 'Story not found' });
    }

    // Create a new therapy
    const therapy = new Therapy({ storyId, title, chapters });
    await therapy.save();

    res.status(201).json({ message: 'Therapy created successfully', therapy });
  } catch (error) {
    res.status(500).json({ message: 'Error creating therapy', error });
  }
});

// Fetch therapy by story ID
router.get('/:storyId', async (req, res) => {
  try {
    const therapy = await Therapy.findOne({ storyId: req.params.storyId });
    if (!therapy) {
      return res.status(404).json({ message: 'Therapy not found' });
    }
    res.json(therapy);
  } catch (error) {
    res.status(500).json({ message: 'Error fetching therapy data', error });
  }
});

// Update a therapy by story ID
router.put('/:storyId', async (req, res) => {
  const { title, chapters } = req.body;

  try {
    const therapy = await Therapy.findOneAndUpdate(
      { storyId: req.params.storyId },
      { title, chapters },
      { new: true } // Return the updated document
    );

    if (!therapy) {
      return res.status(404).json({ message: 'Therapy not found' });
    }

    res.json({ message: 'Therapy updated successfully', therapy });
  } catch (error) {
    res.status(500).json({ message: 'Error updating therapy', error });
  }
});

// Delete a therapy by story ID
router.delete('/:storyId', async (req, res) => {
  try {
    const therapy = await Therapy.findOneAndDelete({ storyId: req.params.storyId });
    if (!therapy) {
      return res.status(404).json({ message: 'Therapy not found' });
    }

    res.json({ message: 'Therapy deleted successfully' });
  } catch (error) {
    res.status(500).json({ message: 'Error deleting therapy', error });
  }
});

module.exports = router;
