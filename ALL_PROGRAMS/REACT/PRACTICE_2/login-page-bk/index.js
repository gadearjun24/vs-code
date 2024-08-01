const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const userRouter = require("./routers/users");
require("dotenv").config();

const server = express();




// middle ware =>
server.use(cors());
server.use(express.json());
server.use("/users", userRouter.router);
// mongoose connection=>

function mongooseConnection() {
  mongoose
    .connect(process.env.MONGODB_URL)
    .then(() => {
      console.log("connected to mongoose");
    })
    .catch((err) => {
      console.log({ err });
    });
}
mongooseConnection();

// server connection =>
server.listen(process.env.PORT, () => {
  console.log(`server connected to ${process.env.PORT}  port`);
});
