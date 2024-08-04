import { useNavigate } from "react-router-dom";
import "../App.css";

function Home() {
  const navigate = useNavigate();

  function handleLoginClick() {
    navigate("/login");
  }
  function handleSignUpClick() {
    navigate("/signup");
  }
  function handleDashboardClick() {
    navigate("/dashboard");
  }

  return (
    <>
      <center>
        <h1>WELCOME</h1>
        <br />
        <br />
        <button onClick={handleLoginClick}>Login</button>
        <prev> </prev>
        <button onClick={handleSignUpClick}>Sign Up</button>
        <prev> </prev>
        <button onClick={handleDashboardClick}>Dashboard</button>
      </center>
    </>
  );
}

export default Home;
