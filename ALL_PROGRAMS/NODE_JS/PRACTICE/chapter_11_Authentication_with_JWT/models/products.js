const { Schema, mongoose } = require("mongoose");

const productSchema = new Schema({
  title: { type: String },
  description: { type: String },
  rating: { type: Number },
  discount: { type: Number },
  price: { type: Number },
});

const Product = mongoose.model("product", productSchema);

exports.Product = Product;
