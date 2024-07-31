const express = require("express");
const userController = require("../controllers/users");

const router = express.Router();

router
  .get("/", userController.getAllUsers)
  .get("/:id", userController.getOneUser)
  .post("/", userController.createNewUser)
  .patch("/:id", userController.updateUser)
  .delete("/:id", userController.deleteUser);

exports.router = router;
