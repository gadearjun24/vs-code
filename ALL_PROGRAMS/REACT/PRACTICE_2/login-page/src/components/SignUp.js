import { useNavigate } from "react-router-dom";
import "./SignUp.css";
import { useState } from "react";
import axios from "axios";

function SignUp() {
  async function handelSignUpCheck() {
    try {
      var res = await axios.post(
        "https://ominous-disco-r4g77g74rjrqhwqq6-8080.app.github.dev/users/signup",
        info,
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (res.status === 200) {
        console.log(res.data);
        navigate("/login");
      }
    } catch (err) {
      console.log({ err });
    }
  }

  const navigate = useNavigate();

  const [info, setInfo] = useState({});

  function handelOnChange(e) {
    setInfo({ ...info, [e.target.name]: e.target.value });
  }

  function handleSignUpClick(e) {
    navigate("/login");
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
          <button onClick={handelSignUpCheck}>Sign Up</button>
          <label onClick={handleSignUpClick}>already you have account?</label>
        </div>
      </div>
    </>
  );
}
export default SignUp;
