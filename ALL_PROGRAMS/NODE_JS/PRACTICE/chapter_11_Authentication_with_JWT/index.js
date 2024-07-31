require("dotenv").config();
const express = require("express");

const server = express();

// Middle-wares=>

// server connection =>
try {
  server.listen(process.env.PORT || 8080, () => {
    console.log(`server is running on ${process.env.PORT} port`);
  });
} catch (err) {
  console.log({ err });
}
