const express = require("express");
const userController = require("../controllers/users");

const router = express.Router();

router
  .get("/", userController.get)
  .post("/signup", userController.signUp)
  .post("/login", userController.login);

exports.router = router;
