const express = require('express');
const { saveResult } = require('../controllers/resultController'); // Import the controller function
const router = express.Router();

router.post('/', saveResult); // Define POST route

module.exports = router;
