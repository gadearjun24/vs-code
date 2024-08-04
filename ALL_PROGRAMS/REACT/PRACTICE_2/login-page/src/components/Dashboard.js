import "./Dashboard.css";
import { useNavigate } from "react-router-dom";

function Dashboard({ verify, fetchData }) {
  fetchData();

  const navigate = useNavigate();
  function handleLoginClick() {
    navigate("/login");
  }
  function handleLogoutClick() {
    localStorage.setItem("token", "");
    fetchData();
  }

  if (verify) {
    return (
      <>
        <center>
          <h1>DASHBOARD</h1>
          <br />
          <br />
          <button onClick={handleLogoutClick}>Logout</button>
        </center>
      </>
    );
  } else {
    return (
      <>
        <center>
          <button onClick={handleLoginClick}>Login</button>
        </center>
      </>
    );
  }
}
export default Dashboard;
