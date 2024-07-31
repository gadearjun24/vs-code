const express = require("express");
const productController = require("../controllers/products");

const router = express.Router();

router
  .get("/", productController.getAllProducts)
  .get("/:id", productController.getOneProduct)
  .post("/", productController.createProduct)
  .delete("/:id", productController.deleteProduct)
  .patch("/:id", productController.updateProduct);

exports.router = router;
