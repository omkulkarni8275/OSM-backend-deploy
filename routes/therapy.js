const express = require('express');
const Therapy = require('../models/Therapy');

const router = express.Router();

// Fetch therapy based on story ID
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

module.exports = router;
