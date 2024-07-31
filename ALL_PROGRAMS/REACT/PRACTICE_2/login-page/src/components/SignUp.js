import "./SignUp.css";
import { useState } from "react";

function SignUp() {
  const [info, setInfo] = useState({});
  const [values, setValues] = useState({});

  function handelOnChange(e) {
    setInfo({ ...info, [e.target.name]: e.target.value });
  }

  function handelClick() {
    setValues(info);
    console.log(values);
  }

  return (
    <>
      <div className="container">
        <label>Email : </label>
        <input
          onChange={handelOnChange}
          name="email"
          type="email"
          placeholder="Enter email"
        ></input>
        <br />
        <label>Password : </label>
        <input
          onChange={handelOnChange}
          name="password"
          type="password"
          placeholder="Enter the password"
        ></input>
        <div className="btn">
          <button onClick={handelClick}>Sign Up</button>
        </div>
      </div>
    </>
  );
}
export default SignUp;
