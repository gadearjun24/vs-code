require("dotenv").config();
const express = require("express");
const mongoose = require("mongoose");
const productRouter = require("./routers/products");
const userRouter = require("./routers/users");
const cors = require("cors");

const server = express();

// Middle-wares=>
server.use(cors());
server.use(express.json());
server.use("/api/products/", productRouter.router);
server.use("/api/users/", userRouter.router);

// mongoose connection =>
function mongooseConnection() {
  mongoose
    .connect(process.env.MONGODB_URL)
    .then(() => {
      console.log("mongoose connect succefully");
    })
    .catch((err) => {
      console.log("error occured durring mongoose connection : ", { err });
    });
}
mongooseConnection();

// server connection =>
try {
  server.listen(process.env.PORT || 8080, () => {
    console.log(`server is running on ${process.env.PORT} port`);
  });
} catch (err) {
  console.log({ err });
}
