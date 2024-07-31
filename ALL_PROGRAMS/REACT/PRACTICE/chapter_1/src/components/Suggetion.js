import "./Search.css";

function Suggetion({ suggetion }) {
  return (
    <>
      {console.log("sugetionnnnnnnnn", suggetion)}

      <div className="suggetion">
        {suggetion.map((ele) => {
          return <span>{ele}</span>;
        })}
      </div>
    </>
  );
}
export default Suggetion;
