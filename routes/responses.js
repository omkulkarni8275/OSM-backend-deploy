// routes/assessments.js
const express = require('express');
const router = express.Router();
const Response = require('../models/Response');

// POST route for assessments
// Route to save the response data
router.post('/', async (req, res) => {
  try {
    // Extract data from the request body
    const {
      gender,
      age,
      personality,
      handedness,
      illnessFear,
      deathFear,
      nightStartled,
      sleepHours,
      hoarding,
      repetitiveActions,
      sequenceRestless,
      avoidTouch,
      thoughtControl,
      checkThings,
      itemsArranged,
      handWashing,
      engrossedThoughts,
      checkingGas,
      repulsiveThoughts,
    } = req.body;

    // Create a new response object using the model
    const newResponse = new Response({
      gender,
      age,
      personality,
      handedness,
      illnessFear,
      deathFear,
      nightStartled,
      sleepHours,
      hoarding,
      repetitiveActions,
      sequenceRestless,
      avoidTouch,
      thoughtControl,
      checkThings,
      itemsArranged,
      handWashing,
      engrossedThoughts,
      checkingGas,
      repulsiveThoughts,
    });

    // Save the response to the database
    await newResponse.save();

    // Send success response
    res.status(201).json({ message: 'Response saved successfully', data: newResponse });
  } catch (error) {
    console.error(`Error saving assessment data: ${error}`);
    res.status(500).json({ message: 'Failed to save response', error });
  }
});

module.exports = router;
