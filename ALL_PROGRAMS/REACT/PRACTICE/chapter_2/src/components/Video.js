import "./Video.css";

function Video({ title, views, verified, image, channel_name }) {
    
  function handelVideoClick() {
    console.log("title : ", title, "\n", "channel name : ", channel_name);
  }

  return (
    <>
      <div className="container" onClick={handelVideoClick}>
        <div className="image">
          <img src={image} alt="error" />
        </div>
        <div className="information">
          <p>Title :{`${title}. ${verified === "true" ? "☑️" : ""} `}</p>
          <p>Views : {views}</p>
          <p>Channel :{channel_name} </p>
        </div>
      </div>
    </>
  );
}

export default Video;
