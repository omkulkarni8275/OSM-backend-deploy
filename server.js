// server.js
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const connectDB = require('./db');
const assessmentsRoute = require('./routes/responses');
const storiesRoute = require('./routes/stories');
const contactRoute = require('./routes/contactroute');
const therapiesRoute = require('./routes/therapy');
const resultRoutes = require('./routes/resultRoutes');
const app = express();
const port = process.env.PORT || 7000;

// Connect to MongoDB
connectDB()
  .then(() => console.log('MongoDB connected successfully'))
  .catch(err => console.error('MongoDB connection error:', err));

app.use(cors());
app.use(bodyParser.json());

// Detailed request logging middleware
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  console.log('Request Headers:', req.headers);
  console.log('Request Body:', req.body);
  next();
});

app.use('/api/resultstorage', resultRoutes);
app.use('/api/responses', assessmentsRoute);
app.use('/api/stories', storiesRoute); 
app.use("/api", contactRoute);
app.use('/api/therapy', therapiesRoute);


// Catch-all route for debugging
app.use('*', (req, res) => {
  console.log(`[${new Date().toISOString()}] 404 - Route not found: ${req.method} ${req.url}`);
  res.status(404).send('Route not found');
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(`[${new Date().toISOString()}] Error:`, err);
  res.status(500).send('Something went wrong!');
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});




