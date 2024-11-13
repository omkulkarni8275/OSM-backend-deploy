const express = require('express');
const contactController = require('../contactController');
const router = express.Router();

// POST route for the contact form
router.post("/contact", (req, res) => {
    console.log("POST request received for /contact");
    contactController.contactUs(req, res);
});

module.exports = router;
